"""DFS-HC Phase 2: hub completion for many-to-many peer paths.

After classical DFS finishes hub ``H``, synthesize::

    MEAN(B.D.MEAN(B.H.f))

for each true M2M bridge ``H ← B → D`` and each **hub arm** ``f``
(already emitted by DFS on ``H``). No prefer-list / semantic priority.

Hub arms are selected from the classical-DFS feature matrix (shallow
hub identities + 1-hop aggs). Default backend is FastDFS ``dfs2sql``
(DuckDB); set ``RDBLEARN_DFS_HC_BACKEND=pandas`` for the legacy path.

See ``results/peer_aware_dfs_algorithm.md``.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_AGG = ("COUNT", "MIN", "MAX", "MEAN", "MEDIAN", "STD", "SUM", "MODE")
_ARM_COUNT = re.compile(r"^COUNT\(([A-Za-z_][A-Za-z0-9_]*)\)$")
_ARM_AGG = re.compile(
    r"^(MIN|MAX|MEAN|MEDIAN|STD|SUM|MODE)\(([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\)$"
)
_ARM_ID = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)$")


@dataclass(frozen=True)
class BridgeSpec:
    table: str
    hub_fk: str
    dim_fk: str
    dim_name: str
    time_col: str | None


def _parse_hub(key_mappings: dict[str, str]) -> tuple[str, str, str] | None:
    for x_col, ref in key_mappings.items():
        if isinstance(ref, str) and "." in ref:
            table, pk = ref.split(".", 1)
            return x_col, table, pk
    return None


def _time_col(rdb, table: str) -> str | None:
    try:
        return rdb.get_table_metadata(table).time_column
    except Exception:
        return None


def discover_bridges(rdb, hub_table: str, hub_pk: str) -> list[BridgeSpec]:
    """M2M bridges H←B→D where D is *not* itself a direct child of H.

    Excludes diamond edges like studies←outcome_analyses→outcomes.
    """
    rels = rdb.get_relationships()
    hub_children = {
        child for child, _, parent, parent_col in rels
        if parent == hub_table and parent_col == hub_pk
    }
    out: list[BridgeSpec] = []
    seen: set[tuple[str, str]] = set()
    for child, child_col, parent, parent_col in rels:
        if parent != hub_table or parent_col != hub_pk:
            continue
        for c2, ccol2, p2, _pcol2 in rels:
            if c2 != child or p2 == hub_table:
                continue
            if p2 in hub_children:
                continue  # not a peer dimension
            key = (child, p2)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                BridgeSpec(
                    table=child,
                    hub_fk=child_col,
                    dim_fk=ccol2,
                    dim_name=p2,
                    time_col=_time_col(rdb, child),
                )
            )
    return out


def _strip_feat_name(col: str, hub_table: str) -> str:
    s = str(col)
    if "__" in s:
        s = s.split("__", 1)[1]
    prefix = f"{hub_table}."
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


def select_hub_arm_names(
    feat_columns: list[str],
    hub_table: str,
    bridge_tables: set[str],
    *,
    rdb: Any = None,
) -> list[str]:
    """Shallow hub-arm features from classical DFS, in column order (no prefer-list).

    Kept:
      - numeric hub identities (dtype-checked when ``rdb`` given)
      - 1-hop ``COUNT(child)`` / ``AGG(child.col)``
    Dropped:
      - keys / timestamps used as join context
      - categoricals / non-numeric identities
      - features whose child table is a peer-bridge table
      - deeper stacks
    """
    skip_idents = {
        "timestamp",
        "nct_id",
        hub_table,
    }
    numeric_idents: set[str] | None = None
    if rdb is not None:
        hub_df = rdb.get_table(hub_table)
        numeric_idents = {
            c
            for c in hub_df.columns
            if pd.api.types.is_numeric_dtype(hub_df[c])
        }

    arms: list[str] = []
    seen: set[str] = set()
    for col in feat_columns:
        name = _strip_feat_name(col, hub_table)
        if name in seen:
            continue
        m_count = _ARM_COUNT.match(name)
        m_agg = _ARM_AGG.match(name)
        m_id = _ARM_ID.match(name)
        if m_count:
            child = m_count.group(1)
            if child in bridge_tables:
                continue
            arms.append(name)
            seen.add(name)
        elif m_agg:
            child = m_agg.group(2)
            if child in bridge_tables:
                continue
            arms.append(name)
            seen.add(name)
        elif m_id:
            ident = m_id.group(1)
            if ident in skip_idents or ident.endswith("_id") or ident.endswith("Id"):
                continue
            if ident.endswith("_epochtime"):
                continue
            if numeric_idents is not None and ident not in numeric_idents:
                continue
            arms.append(name)
            seen.add(name)
    return arms


def _child_frame(rdb, child: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    cdf = rdb.get_table(child)
    tcol = _time_col(rdb, child)
    if tcol and tcol in cdf.columns:
        return cdf.loc[pd.to_datetime(cdf[tcol]) < cutoff]
    return cdf


def compute_arm_series(
    rdb,
    hub_table: str,
    hub_pk: str,
    arm_name: str,
    cutoff: pd.Timestamp,
) -> pd.Series | None:
    """Recompute one hub arm on all hub entities at ``cutoff``."""
    m_count = _ARM_COUNT.match(arm_name)
    if m_count:
        child = m_count.group(1)
        c_ok = _child_frame(rdb, child, cutoff)
        # FK col to hub
        rels = rdb.get_relationships()
        fk = next(
            (
                cc
                for c, cc, p, pc in rels
                if c == child and p == hub_table and pc == hub_pk
            ),
            None,
        )
        if fk is None or fk not in c_ok.columns:
            return None
        return c_ok.groupby(fk, sort=False).size().astype("float64")

    m_agg = _ARM_AGG.match(arm_name)
    if m_agg:
        op, child, col = m_agg.group(1), m_agg.group(2), m_agg.group(3)
        c_ok = _child_frame(rdb, child, cutoff)
        rels = rdb.get_relationships()
        fk = next(
            (
                cc
                for c, cc, p, pc in rels
                if c == child and p == hub_table and pc == hub_pk
            ),
            None,
        )
        if fk is None or fk not in c_ok.columns or col not in c_ok.columns:
            return None
        g = c_ok.groupby(fk, sort=False)[col]
        opmap = {
            "MIN": g.min,
            "MAX": g.max,
            "MEAN": g.mean,
            "MEDIAN": g.median,
            "STD": g.std,
            "SUM": g.sum,
            "MODE": lambda: g.agg(
                lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
            ),
        }
        fn = opmap.get(op)
        if fn is None:
            return None
        return pd.to_numeric(fn(), errors="coerce")

    m_id = _ARM_ID.match(arm_name)
    if m_id:
        col = m_id.group(1)
        hub_df = rdb.get_table(hub_table)
        if col not in hub_df.columns:
            return None
        hub_time = _time_col(rdb, hub_table)
        hub_ok = hub_df
        if hub_time and hub_time in hub_df.columns:
            hub_ok = hub_df.loc[pd.to_datetime(hub_df[hub_time]) < cutoff]
        s = hub_ok.set_index(hub_pk)[col]
        return pd.to_numeric(s, errors="coerce")

    return None


def _as_join_key(s: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(s).astype("string")


def _peer_for_bridge_aligned(
    bridge_df: pd.DataFrame,
    hub_fk: str,
    dim_fk: str,
    arm: pd.Series,
    hub_ids: np.ndarray,
    row_index: pd.Index,
) -> pd.Series:
    b = bridge_df[[hub_fk, dim_fk]].copy()
    b[hub_fk] = _as_join_key(b[hub_fk])
    b[dim_fk] = _as_join_key(b[dim_fk])
    arm_s = arm.copy()
    arm_s.index = _as_join_key(arm_s.index).to_numpy()
    b["f"] = b[hub_fk].map(arm_s)
    dim_mean = b.groupby(dim_fk, sort=False)["f"].mean()
    tmp = pd.DataFrame(
        {"_row": np.arange(len(hub_ids)), hub_fk: _as_join_key(hub_ids)}
    )
    mine = tmp.merge(b[[hub_fk, dim_fk]], on=hub_fk, how="left")
    mine["peer"] = mine[dim_fk].map(dim_mean)
    agg = mine.groupby("_row", sort=False)["peer"].mean()
    return pd.Series(agg.reindex(np.arange(len(hub_ids))).to_numpy(), index=row_index)


def peer_feature_name(bridge: BridgeSpec, arm_name: str, hub_table: str) -> str:
    return (
        f"MEAN({bridge.table}.{bridge.dim_name}."
        f"MEAN({bridge.table}.{hub_table}.{arm_name}))"
    )


def _plan_peers(
    rdb,
    feats: pd.DataFrame,
    X: pd.DataFrame,
    key_mappings: dict[str, str],
    *,
    max_peer_features: int,
) -> tuple[str, str, str, list[BridgeSpec], list[str], list[tuple[BridgeSpec, str]]] | None:
    hub = _parse_hub(key_mappings)
    if hub is None:
        logger.warning("DFS-HC: no hub key_mappings; skipping")
        return None
    x_col, hub_table, hub_pk = hub
    if x_col not in X.columns:
        logger.warning("DFS-HC: hub column {} missing on X; skipping", x_col)
        return None
    bridges = discover_bridges(rdb, hub_table, hub_pk)
    allow = os.environ.get("RDBLEARN_DFS_HC_BRIDGES", "").strip()
    if allow:
        allow_set = {x.strip() for x in allow.split(",") if x.strip()}
        bridges = [b for b in bridges if b.table in allow_set]
        logger.info(
            "DFS-HC: bridge filter {!r} → {}",
            sorted(allow_set),
            [b.table for b in bridges],
        )
    if not bridges:
        logger.info(
            "DFS-HC: no M2M bridges for hub {}.{}; skipping", hub_table, hub_pk
        )
        return None
    bridge_tables = {b.table for b in bridges}
    arm_names_raw = select_hub_arm_names(
        list(feats.columns), hub_table, bridge_tables, rdb=rdb
    )
    # Drop hub-arm names that are not raw RDB columns (e.g. *.date_epochtime).
    arm_names = [
        a for a in arm_names_raw if _arm_recomputable_on_rdb(rdb, hub_table, a)
    ]
    n_drop = len(arm_names_raw) - len(arm_names)
    if n_drop:
        logger.info(
            "DFS-HC: dropped {} non-RDB hub arms (kept {})",
            n_drop,
            len(arm_names),
        )
    arm_allow = os.environ.get("RDBLEARN_DFS_HC_ARMS", "").strip()
    if arm_allow:
        # Comma-separated exact arm names, or "specialized" preset.
        if arm_allow.lower() == "specialized":
            allow_arms = [
                "MIN(outcome_analyses.p_value)",
                "MEAN(outcome_analyses.p_value)",
                "MAX(outcome_analyses.p_value)",
                "STD(outcome_analyses.p_value)",
                "COUNT(outcome_analyses)",
                "enrollment",
            ]
        else:
            allow_arms = [x.strip() for x in arm_allow.split(",") if x.strip()]
        allow_set = set(allow_arms)
        arm_names = [a for a in arm_names if a in allow_set]
        # Preserve specialized order when preset is used.
        if arm_allow.lower() == "specialized":
            arm_names = [a for a in allow_arms if a in set(arm_names)]
        logger.info(
            "DFS-HC: arm filter → {} arms {}",
            len(arm_names),
            arm_names,
        )
    if not arm_names:
        logger.info("DFS-HC: no hub arms on {}; skipping", hub_table)
        return None
    planned: list[tuple[BridgeSpec, str]] = []
    for br in bridges:
        for arm_name in arm_names:
            planned.append((br, arm_name))
            if len(planned) >= max_peer_features:
                break
        if len(planned) >= max_peer_features:
            break
    return x_col, hub_table, hub_pk, bridges, arm_names, planned


def _log_phase2(
    hub_table: str,
    hub_pk: str,
    bridges: list[BridgeSpec],
    arm_names: list[str],
    out: pd.DataFrame,
    skipped_arms: set[str],
    *,
    backend: str,
) -> None:
    n_valid = int(out.notna().any(axis=1).sum()) if len(out.columns) else 0
    logger.info(
        "DFS-HC Phase 2 ({}): hub={}.{} bridges={} hub_arms={} peer_cols={} "
        "skipped_arms={} rows_with_any_valid={}/{}",
        backend,
        hub_table,
        hub_pk,
        [b.table for b in bridges],
        len(arm_names),
        out.shape[1],
        len(skipped_arms),
        n_valid,
        len(out),
    )
    if skipped_arms:
        logger.info(
            "DFS-HC skipped unrecomputable hub arms (sample): {}",
            sorted(skipped_arms)[:12],
        )
    key = (
        f"MEAN(conditions_studies.conditions.MEAN("
        f"conditions_studies.{hub_table}.MIN(outcome_analyses.p_value)))"
    )
    if key in out.columns:
        logger.info("DFS-HC includes key peer feature: {}", key)
    else:
        logger.warning("DFS-HC MISSING key peer feature: {}", key)


def _hub_completion_features_pandas(
    rdb,
    X: pd.DataFrame,
    x_col: str,
    hub_table: str,
    hub_pk: str,
    bridges: list[BridgeSpec],
    arm_names: list[str],
    planned: list[tuple[BridgeSpec, str]],
    cutoff_time_column: str,
) -> pd.DataFrame:
    keys = pd.DataFrame(
        {
            "hub_id": X[x_col],
            "cutoff": pd.to_datetime(X[cutoff_time_column]),
        },
        index=X.index,
    )
    col_names = [peer_feature_name(br, a, hub_table) for br, a in planned]
    out = pd.DataFrame(np.nan, index=X.index, columns=col_names, dtype="float64")
    bridge_frames = {br.table: rdb.get_table(br.table) for br in bridges}
    used_arms = sorted({a for _, a in planned})
    skipped_arms: set[str] = set()
    for cutoff, grp in keys.groupby("cutoff", sort=False):
        arms: dict[str, pd.Series] = {}
        for arm_name in used_arms:
            s = compute_arm_series(rdb, hub_table, hub_pk, arm_name, cutoff)
            if s is None:
                skipped_arms.add(arm_name)
                continue
            arms[arm_name] = s
        hub_ids = grp["hub_id"].to_numpy()
        filtered: dict[str, pd.DataFrame] = {}
        for br in bridges:
            bdf = bridge_frames[br.table]
            if br.time_col and br.time_col in bdf.columns:
                filtered[br.table] = bdf.loc[pd.to_datetime(bdf[br.time_col]) < cutoff]
            else:
                filtered[br.table] = bdf
        for br, arm_name in planned:
            if arm_name not in arms:
                continue
            s = _peer_for_bridge_aligned(
                filtered[br.table],
                br.hub_fk,
                br.dim_fk,
                arms[arm_name],
                hub_ids,
                grp.index,
            )
            out.loc[grp.index, peer_feature_name(br, arm_name, hub_table)] = (
                s.to_numpy()
            )
    _log_phase2(
        hub_table, hub_pk, bridges, arm_names, out, skipped_arms, backend="pandas"
    )
    return out


def _es_rel(es, child: str, parent: str):
    for r in es.relationships:
        if (
            r.child_dataframe.ww.name == child
            and r.parent_dataframe.ww.name == parent
        ):
            return r
    raise KeyError(f"relationship {child} -> {parent}")


def _arm_recomputable_on_rdb(rdb, hub_table: str, arm_name: str) -> bool:
    """True if arm references raw RDB columns (not temporal-diff derived names)."""
    m = _ARM_COUNT.match(arm_name)
    if m:
        return m.group(1) in rdb.table_names
    m = _ARM_AGG.match(arm_name)
    if m:
        child, col = m.group(2), m.group(3)
        if child not in rdb.table_names:
            return False
        return col in rdb.get_table(child).columns
    m = _ARM_ID.match(arm_name)
    if m:
        return m.group(1) in rdb.get_table(hub_table).columns
    return False


def _build_arm_feature(es, hub_table: str, arm_name: str):
    """Build a Featuretools feature on ``hub_table`` for a hub-arm name."""
    from featuretools import AggregationFeature, IdentityFeature
    from featuretools.entityset.relationship import RelationshipPath
    from featuretools.primitives import (
        Count,
        Max,
        Mean,
        Median,
        Min,
        Mode,
        Std,
        Sum,
    )

    prims = {
        "MIN": Min(),
        "MAX": Max(),
        "MEAN": Mean(),
        "MEDIAN": Median(),
        "STD": Std(),
        "SUM": Sum(),
        "MODE": Mode(),
    }
    m = _ARM_COUNT.match(arm_name)
    if m:
        child = m.group(1)
        idx = es[child].ww.index
        return AggregationFeature(
            IdentityFeature(es[child].ww[idx]),
            parent_dataframe_name=hub_table,
            primitive=Count(),
            relationship_path=RelationshipPath(
                [(False, _es_rel(es, child, hub_table))]
            ),
        )
    m = _ARM_AGG.match(arm_name)
    if m:
        op, child, col = m.group(1), m.group(2), m.group(3)
        if col not in es[child].columns:
            return None
        return AggregationFeature(
            IdentityFeature(es[child].ww[col]),
            parent_dataframe_name=hub_table,
            primitive=prims[op],
            relationship_path=RelationshipPath(
                [(False, _es_rel(es, child, hub_table))]
            ),
        )
    m = _ARM_ID.match(arm_name)
    if m:
        col = m.group(1)
        if col not in es[hub_table].columns:
            return None
        return IdentityFeature(es[hub_table].ww[col])
    return None


def _build_peer_feature(es, hub_table: str, bridge: BridgeSpec, arm_name: str):
    """``Direct( MEAN(B.D.MEAN(B.H.arm)) → __target__ )``."""
    from featuretools import AggregationFeature, DirectFeature
    from featuretools.entityset.relationship import RelationshipPath
    from featuretools.primitives import Mean

    arm = _build_arm_feature(es, hub_table, arm_name)
    if arm is None:
        return None
    d1 = DirectFeature(
        arm, bridge.table, relationship=_es_rel(es, bridge.table, hub_table)
    )
    agg_dim = AggregationFeature(
        d1,
        parent_dataframe_name=bridge.dim_name,
        primitive=Mean(),
        relationship_path=RelationshipPath(
            [(False, _es_rel(es, bridge.table, bridge.dim_name))]
        ),
    )
    d2 = DirectFeature(
        agg_dim,
        bridge.table,
        relationship=_es_rel(es, bridge.table, bridge.dim_name),
    )
    peer_h = AggregationFeature(
        d2,
        parent_dataframe_name=hub_table,
        primitive=Mean(),
        relationship_path=RelationshipPath(
            [(False, _es_rel(es, bridge.table, hub_table))]
        ),
    )
    return DirectFeature(
        peer_h, "__target__", relationship=_es_rel(es, "__target__", hub_table)
    )


def _strip_synthetic_index(rdb) -> None:
    """Remove leftover ``__index__`` columns that EntitySet builders may inject."""
    for name in list(rdb.table_names):
        df = rdb.get_table(name)
        if "__index__" in getattr(df, "columns", ()):
            df.drop(columns=["__index__"], inplace=True)


def _hub_completion_features_sql(
    rdb,
    X: pd.DataFrame,
    x_col: str,
    hub_table: str,
    hub_pk: str,
    bridges: list[BridgeSpec],
    arm_names: list[str],
    planned: list[tuple[BridgeSpec, str]],
    cutoff_time_column: str,
) -> pd.DataFrame:
    """Compile peer stacks to Featuretools features and execute via dfs2sql."""
    from fastdfs.dfs.base_engine import DFSConfig
    from fastdfs.dfs.dfs2sql_engine import DFS2SQLEngine

    t0 = time.perf_counter()
    _strip_synthetic_index(rdb)
    try:
        cfg = DFSConfig(engine="dfs2sql", use_cutoff_time=True, include_cutoff_time=False)
        engine = DFS2SQLEngine(cfg)
        es = engine._build_entity_set_from_rdb(rdb)

        target = pd.DataFrame(
            {
                "__target_index__": np.arange(len(X), dtype=np.int64),
                x_col: X[x_col].astype("string").to_numpy(),
                cutoff_time_column: pd.to_datetime(X[cutoff_time_column]).to_numpy(),
            }
        )
        # Align join key dtype with hub PK for EntitySet relationship.
        hub_df = rdb.get_table(hub_table)
        if hub_pk in hub_df.columns:
            # EntitySet will coerce child→parent; keep string for DuckDB join safety.
            pass
        es = es.add_dataframe(
            dataframe_name="__target__",
            dataframe=target,
            index="__target_index__",
            time_index=cutoff_time_column,
        )
        es = es.add_relationship(
            parent_dataframe_name=hub_table,
            parent_column_name=hub_pk,
            child_dataframe_name="__target__",
            child_column_name=x_col,
        )

        ft_features = []
        col_names: list[str] = []
        skipped_arms: set[str] = set()
        for br, arm_name in planned:
            try:
                feat = _build_peer_feature(es, hub_table, br, arm_name)
            except Exception as e:
                logger.debug("DFS-HC SQL skip arm {} on {}: {}", arm_name, br.table, e)
                skipped_arms.add(arm_name)
                continue
            if feat is None:
                skipped_arms.add(arm_name)
                continue
            ft_features.append(feat)
            col_names.append(peer_feature_name(br, arm_name, hub_table))

        if not ft_features:
            out = pd.DataFrame(index=X.index)
            _log_phase2(
                hub_table, hub_pk, bridges, arm_names, out, skipped_arms, backend="sql"
            )
            return out

        logger.info(
            "DFS-HC SQL: compiling {} peer features via dfs2sql (EntitySet+DuckDB)…",
            len(ft_features),
        )
        fm = engine.compute_feature_matrix(
            rdb=rdb,
            target_dataframe=target,
            key_mappings={x_col: f"{hub_table}.{hub_pk}"},
            cutoff_time_column=cutoff_time_column,
            features=ft_features,
            config=cfg,
        )
        # Map FT names → canonical peer names (order preserved).
        value_cols = [c for c in fm.columns if c != "__target_index__"]
        if len(value_cols) != len(col_names):
            logger.warning(
                "DFS-HC SQL: expected {} value cols, got {} ({})",
                len(col_names),
                len(value_cols),
                value_cols[:5],
            )
        data = {}
        for i, name in enumerate(col_names):
            if i >= len(value_cols):
                data[name] = np.full(len(X), np.nan, dtype="float64")
            else:
                data[name] = pd.to_numeric(fm[value_cols[i]].to_numpy(), errors="coerce")
        out = pd.DataFrame(data, index=X.index)
        logger.info(
            "DFS-HC SQL finished in {:.1f}s ({} peers)",
            time.perf_counter() - t0,
            out.shape[1],
        )
        _log_phase2(
            hub_table, hub_pk, bridges, arm_names, out, skipped_arms, backend="sql"
        )
        return out
    finally:
        # Belt-and-suspenders: never leave synthetic indexes on the live RDB
        # (breaks later DFS canonicalize_key_types / metadata checks).
        _strip_synthetic_index(rdb)


def hub_completion_features(
    rdb,
    feats: pd.DataFrame,
    X: pd.DataFrame,
    key_mappings: dict[str, str],
    cutoff_time_column: str,
    *,
    max_peer_features: int = 256,
    backend: str | None = None,
) -> pd.DataFrame:
    planned_pack = _plan_peers(
        rdb, feats, X, key_mappings, max_peer_features=max_peer_features
    )
    if planned_pack is None:
        return pd.DataFrame(index=X.index)
    x_col, hub_table, hub_pk, bridges, arm_names, planned = planned_pack

    be = (backend or os.environ.get("RDBLEARN_DFS_HC_BACKEND", "sql")).strip().lower()
    if be not in ("sql", "pandas"):
        logger.warning("DFS-HC: unknown backend {!r}; using sql", be)
        be = "sql"

    if be == "pandas":
        return _hub_completion_features_pandas(
            rdb,
            X,
            x_col,
            hub_table,
            hub_pk,
            bridges,
            arm_names,
            planned,
            cutoff_time_column,
        )
    return _hub_completion_features_sql(
        rdb,
        X,
        x_col,
        hub_table,
        hub_pk,
        bridges,
        arm_names,
        planned,
        cutoff_time_column,
    )


def append_hub_completion(
    feats: pd.DataFrame,
    X: pd.DataFrame,
    cutoff_time_column: str | None,
    *,
    rdb: Any = None,
    key_mappings: dict[str, str] | None = None,
    max_peer_features: int = 256,
    backend: str | None = None,
    **_kwargs,
) -> pd.DataFrame:
    if cutoff_time_column is None or rdb is None or not key_mappings:
        logger.warning("DFS-HC: missing rdb/key_mappings/cutoff; skipping")
        return feats
    extra = hub_completion_features(
        rdb,
        feats,
        X,
        key_mappings,
        cutoff_time_column,
        max_peer_features=max_peer_features,
        backend=backend,
    )
    if extra.shape[1] == 0:
        return feats
    if not feats.index.equals(extra.index) and len(feats) == len(extra):
        extra = extra.set_index(feats.index)
    return pd.concat([feats, extra], axis=1)
