from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from fastdfs import DFSConfig, RDB


def _stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_dataframe(df: pd.DataFrame) -> str:
    hashed_rows = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(hashed_rows).hexdigest()


def _dfs_config_to_dict(config: DFSConfig) -> Dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    return config.dict()


@dataclass
class DFSCacheContext:
    key: str
    requested_depth: int
    effective_max_depth: int


class DFSDiskCache:
    VERSION = "v1"

    def __init__(
        self,
        cache_dir: str,
        max_depth_mode: str = "requested",
        fixed_max_depth: Optional[int] = None,
        rebuild: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.max_depth_mode = max_depth_mode
        self.fixed_max_depth = fixed_max_depth
        self.rebuild = rebuild
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_context(
        self,
        rdb: RDB,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        dfs_config: DFSConfig,
        extra_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> DFSCacheContext:
        requested_depth = int(dfs_config.max_depth)
        effective_max_depth = requested_depth
        if self.max_depth_mode == "fixed_max" and self.fixed_max_depth is not None:
            effective_max_depth = max(requested_depth, int(self.fixed_max_depth))

        key_payload = {
            "version": self.VERSION,
            "rdb_name": rdb.metadata.name,
            "rdb_schema": self._rdb_schema_fingerprint(rdb),
            "target_hash": _hash_dataframe(target_dataframe),
            "key_mappings": key_mappings,
            "cutoff_time_column": cutoff_time_column,
            "dfs_config_without_depth": self._dfs_config_without_depth(dfs_config),
            "extra": extra_fingerprint or {},
        }
        key = _stable_hash(key_payload)
        return DFSCacheContext(
            key=key,
            requested_depth=requested_depth,
            effective_max_depth=effective_max_depth,
        )

    def load_or_compute(
        self,
        context: DFSCacheContext,
        compute_fn: Callable[[int], Tuple[pd.DataFrame, Dict[str, Any]]],
    ) -> pd.DataFrame:
        cached = self._load_best_cached(context)
        if cached is not None and not self.rebuild:
            feature_df, metadata, cache_depth = cached
            logger.info(
                f"DFS cache hit: key={context.key[:10]} depth={cache_depth} requested={context.requested_depth}"
            )
            return self._slice_by_depth(feature_df, metadata, context.requested_depth)

        logger.info(
            f"DFS cache miss: key={context.key[:10]} requested={context.requested_depth}, compute_depth={context.effective_max_depth}"
        )
        return self._compute_and_store(context, compute_fn)

    def _compute_and_store(
        self,
        context: DFSCacheContext,
        compute_fn: Callable[[int], Tuple[pd.DataFrame, Dict[str, Any]]],
    ) -> pd.DataFrame:
        lock_path = self._lock_path(context.key)
        with self._file_lock(lock_path):
            if not self.rebuild:
                cached = self._load_best_cached(context)
                if cached is not None:
                    feature_df, metadata, cache_depth = cached
                    logger.info(
                        f"DFS cache became available while waiting lock: depth={cache_depth} requested={context.requested_depth}"
                    )
                    return self._slice_by_depth(feature_df, metadata, context.requested_depth)

            full_df, metadata = compute_fn(context.effective_max_depth)
            self._validate_payload(full_df, metadata)
            self._write_cache(context, full_df, metadata)
            return self._slice_by_depth(full_df, metadata, context.requested_depth)

    def _validate_payload(self, feature_df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        if "features" not in metadata or "original_columns" not in metadata:
            raise ValueError("DFS metadata missing required fields: features/original_columns")
        feature_cols = {f["feature_name"] for f in metadata["features"]}
        missing = [c for c in feature_cols if c not in feature_df.columns]
        if missing:
            raise ValueError(f"Metadata-feature mismatch, missing columns in matrix: {missing[:5]}")

    def _slice_by_depth(
        self,
        feature_df: pd.DataFrame,
        metadata: Dict[str, Any],
        requested_depth: int,
    ) -> pd.DataFrame:
        original_cols = [c for c in metadata.get("original_columns", []) if c in feature_df.columns]
        selected_features = []
        for feat in metadata.get("features", []):
            name = feat.get("feature_name")
            depth = feat.get("depth")
            if name not in feature_df.columns:
                continue
            if depth is None or int(depth) <= requested_depth:
                selected_features.append(name)
        # Keep deterministic order and avoid duplicates.
        final_cols = list(dict.fromkeys(original_cols + selected_features))
        return feature_df[final_cols]

    def _load_best_cached(
        self, context: DFSCacheContext
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any], int]]:
        root = self.cache_dir / context.key
        if not root.exists():
            return None
        depth_dirs = []
        for p in root.glob("depth_*"):
            if not p.is_dir():
                continue
            try:
                depth = int(p.name.split("_", 1)[1])
            except Exception:
                continue
            if depth >= context.requested_depth:
                depth_dirs.append((depth, p))
        if not depth_dirs:
            return None
        # Prefer the smallest valid depth >= requested to reduce IO.
        depth, selected = sorted(depth_dirs, key=lambda x: x[0])[0]
        feature_df = pd.read_parquet(selected / "feature_matrix.parquet")
        with open(selected / "feature_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return feature_df, metadata, depth

    def _write_cache(self, context: DFSCacheContext, feature_df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        depth_dir = self.cache_dir / context.key / f"depth_{context.effective_max_depth}"
        depth_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "cache_version": self.VERSION,
            "cache_key": context.key,
            "requested_depth": context.requested_depth,
            "stored_depth": context.effective_max_depth,
            "n_rows": int(len(feature_df)),
            "n_cols": int(len(feature_df.columns)),
        }

        self._atomic_write_parquet(feature_df, depth_dir / "feature_matrix.parquet")
        self._atomic_write_json(metadata, depth_dir / "feature_metadata.json")
        self._atomic_write_json(manifest, depth_dir / "manifest.json")

    def _rdb_schema_fingerprint(self, rdb: RDB) -> Dict[str, Any]:
        result: Dict[str, Any] = {"name": rdb.metadata.name, "tables": []}
        for table_name in sorted(rdb.table_names):
            table_meta = rdb.get_table_metadata(table_name)
            df = rdb.get_table(table_name)
            result["tables"].append(
                {
                    "name": table_name,
                    "time_column": table_meta.time_column,
                    "columns": [(c.name, str(c.dtype)) for c in table_meta.columns],
                    "n_rows": int(len(df)),
                    "data_hash": _hash_dataframe(df),
                }
            )
        return result

    def _dfs_config_without_depth(self, dfs_config: DFSConfig) -> Dict[str, Any]:
        cfg = _dfs_config_to_dict(dfs_config)
        cfg.pop("max_depth", None)
        # engine_path is a scratch DuckDB file location; it differs every sweep run
        # (run_sweep_experiment uses a NamedTemporaryFile) and must not affect the key.
        cfg.pop("engine_path", None)
        return cfg

    def _atomic_write_json(self, payload: Dict[str, Any], output_path: Path) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(output_path.parent), delete=False
        ) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, output_path)

    def _atomic_write_parquet(self, df: pd.DataFrame, output_path: Path) -> None:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(output_path.parent), delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output_path)

    def _lock_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.lock"

    def _file_lock(self, lock_path: Path):
        return _FileLock(lock_path)


class _FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fh is None:
            return
        if fcntl is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()
