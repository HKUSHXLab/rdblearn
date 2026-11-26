"""Text-encoder wiring for ``StandardPipeline``.

``text_encoder="tfidf"`` does **not** implement TF-IDF in rdblearn. It:

1. Attaches free-text columns from the entity (anchor) table as pandas ``string``
   dtype (suffixed ``__raw_text``).
2. Asks TabPFN to expand those columns via ``inference_config={"TRANSFORM_TEXT": True}``.

``text_encoder=None`` skips both steps.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastdfs import RDB
from loguru import logger


def parse_entity_key(key_mappings: Dict[str, str]) -> tuple[str, str, str]:
    """Return ``(x_col, entity_table, entity_pk)`` from the first ``table.pk`` mapping."""
    for x_col, ref in key_mappings.items():
        if isinstance(ref, str) and "." in ref:
            table, pk = ref.split(".", 1)
            return x_col, table, pk
    raise ValueError(f"No entity key_mappings of the form table.pk in {key_mappings!r}")


def anchor_text_columns(rdb: RDB, entity_table: str, entity_pk: str) -> List[str]:
    """Object/string columns on ``entity_table``, excluding keys and time."""
    df = rdb.get_table(entity_table)
    meta = rdb.get_table_metadata(entity_table)
    exclude = {entity_pk}
    time_col = getattr(meta, "time_column", None)
    if time_col:
        exclude.add(time_col)
    try:
        for child, child_col, parent, _ in rdb.get_relationships():
            if child == entity_table:
                exclude.add(child_col)
            if parent == entity_table:
                exclude.add(child_col)  # FKs pointing at us still live on child
    except Exception:
        pass
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            cols.append(c)
    return cols


def lookup_anchor_text(
    rdb: RDB,
    X: pd.DataFrame,
    *,
    x_col: str,
    entity_table: str,
    entity_pk: str,
    text_cols: List[str],
) -> pd.DataFrame:
    """Row-aligned text frame for ``X`` via left join on the entity key."""
    if not text_cols:
        return pd.DataFrame(index=range(len(X)))
    adf = rdb.get_table(entity_table)
    available = [c for c in text_cols if c in adf.columns]
    if not available:
        return pd.DataFrame(index=range(len(X)))
    lookup = adf[[entity_pk, *available]].drop_duplicates(subset=[entity_pk]).copy()
    for c in available:
        lookup[c] = lookup[c].map(lambda v: "" if v is None else str(v))
    lookup[entity_pk] = lookup[entity_pk].astype(str)
    raw = X.reset_index(drop=True)[[x_col]].copy()
    raw[x_col] = raw[x_col].astype(str)
    joined = raw.merge(lookup, left_on=x_col, right_on=entity_pk, how="left")
    return joined[available]


def attach_text(df: pd.DataFrame, text: pd.DataFrame | None) -> pd.DataFrame:
    if text is None or text.empty:
        return df
    if len(df) != len(text):
        raise ValueError(f"attach_text: row mismatch df={len(df)} text={len(text)}")
    return pd.concat([df.reset_index(drop=True), text.reset_index(drop=True)], axis=1)


class AnchorTextEncoder:
    """Fit-once anchor text attachment for ``text_encoder='tfidf'``."""

    def __init__(self) -> None:
        self._cols: Optional[List[str]] = None
        self._x_col: Optional[str] = None
        self._entity_table: Optional[str] = None
        self._entity_pk: Optional[str] = None

    def fit(self, rdb: RDB, X: pd.DataFrame, key_mappings: Dict[str, str]) -> "AnchorTextEncoder":
        self._x_col, self._entity_table, self._entity_pk = parse_entity_key(key_mappings)
        self._cols = anchor_text_columns(rdb, self._entity_table, self._entity_pk)
        logger.info(
            "text_encoder=tfidf: {} anchor text cols on {}.{}",
            len(self._cols),
            self._entity_table,
            self._entity_pk,
        )
        return self

    def transform(self, rdb: RDB, feats: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        if not self._cols:
            return feats
        raw = lookup_anchor_text(
            rdb,
            X,
            x_col=self._x_col,  # type: ignore[arg-type]
            entity_table=self._entity_table,  # type: ignore[arg-type]
            entity_pk=self._entity_pk,  # type: ignore[arg-type]
            text_cols=self._cols,
        )
        if raw.empty:
            return feats
        renamed = raw.astype("string").rename(columns={c: f"{c}__raw_text" for c in raw.columns})
        # Drop any prior text attachment so fit/predict never duplicates columns.
        drop = [c for c in feats.columns if c in renamed.columns]
        base = feats.drop(columns=drop) if drop else feats
        return attach_text(base, renamed)


def merge_transform_text(model_params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``model_params`` with TabPFN ``TRANSFORM_TEXT`` enabled."""
    out = dict(model_params)
    ic = dict(out.get("inference_config") or {})
    ic["TRANSFORM_TEXT"] = True
    out["inference_config"] = ic
    return out
