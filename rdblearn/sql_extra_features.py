"""Leakage-safe SQL extra features for RelBench classification tasks.

Features are computed from the original RelBench DB (not the DFS-transformed RDB)
and appended in-memory only — the DFS disk cache is never modified.

All event joins use event_time < cutoff timestamp.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DB_CACHE: Dict[str, Any] = {}
_CON_CACHE: Dict[str, duckdb.DuckDBPyConnection] = {}


def _get_db(dataset: str):
    if dataset not in _DB_CACHE:
        from relbench.datasets import get_dataset

        _DB_CACHE[dataset] = get_dataset(dataset, download=False).get_db(
            upto_test_timestamp=False
        )
    return _DB_CACHE[dataset]


def _get_con(dataset: str) -> duckdb.DuckDBPyConnection:
    if dataset not in _CON_CACHE:
        db = _get_db(dataset)
        con = duckdb.connect()
        for name, table in db.table_dict.items():
            con.register(name, table.df)
        _CON_CACHE[dataset] = con
        logger.info("sql_extra_features: registered RelBench DB {}", dataset)
    return _CON_CACHE[dataset]


def _align_keys(
    X: pd.DataFrame, time_col: str, entity_col: str
) -> Tuple[pd.DataFrame, str]:
    if entity_col not in X.columns:
        raise KeyError(f"Missing entity col {entity_col} in columns={list(X.columns)}")
    if time_col not in X.columns:
        raise KeyError(f"Missing time col {time_col}")
    keys = X[[time_col, entity_col]].copy()
    keys[time_col] = pd.to_datetime(keys[time_col])
    # Normalize time col name to timestamp for SQL templates that expect it
    if time_col != "timestamp":
        keys = keys.rename(columns={time_col: "timestamp"})
        time_sql = "timestamp"
    else:
        time_sql = "timestamp"
    return keys, time_sql


def _run_feature_sql(
    dataset: str,
    keys: pd.DataFrame,
    entity_col: str,
    sql: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    con = _get_con(dataset)
    con.register("target_keys", keys)
    feats = con.execute(sql).df()
    # keys may use 'timestamp' while entity keeps original name
    merge_cols = ["timestamp", entity_col]
    out = keys.merge(feats, on=merge_cols, how="left")
    if len(out) != len(keys):
        raise RuntimeError(f"feature join row mismatch {len(keys)} -> {len(out)}")
    return out[feature_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-task builders
# ---------------------------------------------------------------------------


def _amazon_item_churn(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "product_id"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_review",
        "n_reviews_91d",
        "n_reviews_365d",
        "mean_rating_91d",
        "n_reviews_lifetime",
        "n_distinct_customers_91d",
    ]
    sql = """
    SELECT
      t.timestamp,
      t.product_id,
      date_diff('day', MAX(r.review_time), t.timestamp) AS days_since_last_review,
      COUNT(*) FILTER (
        WHERE r.review_time > t.timestamp - INTERVAL '91 days'
      )::DOUBLE AS n_reviews_91d,
      COUNT(*) FILTER (
        WHERE r.review_time > t.timestamp - INTERVAL '365 days'
      )::DOUBLE AS n_reviews_365d,
      AVG(r.rating) FILTER (
        WHERE r.review_time > t.timestamp - INTERVAL '91 days'
      ) AS mean_rating_91d,
      COUNT(*)::DOUBLE AS n_reviews_lifetime,
      COUNT(DISTINCT r.customer_id) FILTER (
        WHERE r.review_time > t.timestamp - INTERVAL '91 days'
      )::DOUBLE AS n_distinct_customers_91d
    FROM target_keys t
    LEFT JOIN review r
      ON r.product_id = t.product_id AND r.review_time < t.timestamp
    GROUP BY 1, 2
    """
    return _run_feature_sql("rel-amazon", keys, entity, sql, cols)


def _amazon_user_churn(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "customer_id"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_review",
        "n_reviews_91d",
        "n_reviews_365d",
        "mean_rating_91d",
        "n_reviews_lifetime",
        "n_distinct_products_91d",
    ]
    sql = """
    SELECT
      t.timestamp,
      t.customer_id,
      date_diff('day', MAX(r.review_time), t.timestamp) AS days_since_last_review,
      COUNT(*) FILTER (
        WHERE r.review_time > t.timestamp - INTERVAL '91 days'
      )::DOUBLE AS n_reviews_91d,
      COUNT(*) FILTER (
        WHERE r.review_time > t.timestamp - INTERVAL '365 days'
      )::DOUBLE AS n_reviews_365d,
      AVG(r.rating) FILTER (
        WHERE r.review_time > t.timestamp - INTERVAL '91 days'
      ) AS mean_rating_91d,
      COUNT(*)::DOUBLE AS n_reviews_lifetime,
      COUNT(DISTINCT r.product_id) FILTER (
        WHERE r.review_time > t.timestamp - INTERVAL '91 days'
      )::DOUBLE AS n_distinct_products_91d
    FROM target_keys t
    LEFT JOIN review r
      ON r.customer_id = t.customer_id AND r.review_time < t.timestamp
    GROUP BY 1, 2
    """
    return _run_feature_sql("rel-amazon", keys, entity, sql, cols)


def _avito_user_clicks(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "UserID"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_click",
        "n_clicks_4d",
        "n_clicks_14d",
        "n_clicks_28d",
        "n_searches_4d",
        "click_per_search_14d",
        "hist_multiclick_rate",
    ]
    sql = """
    WITH clicks AS (
      SELECT ss.SearchDate AS dt, si.UserID, ss.AdID
      FROM SearchStream ss
      JOIN SearchInfo si ON si.SearchID = ss.SearchID
      WHERE ss.IsClick = 1
    ),
    base AS (
      SELECT
        t.timestamp, t.UserID,
        date_diff('day', MAX(c.dt), t.timestamp) AS days_since_last_click,
        COUNT(*) FILTER (WHERE c.dt > t.timestamp - INTERVAL '4 days')::DOUBLE AS n_clicks_4d,
        COUNT(*) FILTER (WHERE c.dt > t.timestamp - INTERVAL '14 days')::DOUBLE AS n_clicks_14d,
        COUNT(*) FILTER (WHERE c.dt > t.timestamp - INTERVAL '28 days')::DOUBLE AS n_clicks_28d
      FROM target_keys t
      LEFT JOIN clicks c ON c.UserID = t.UserID AND c.dt < t.timestamp
      GROUP BY 1, 2
    ),
    searches AS (
      SELECT
        t.timestamp, t.UserID,
        COUNT(*) FILTER (
          WHERE s.SearchDate > t.timestamp - INTERVAL '4 days'
        )::DOUBLE AS n_searches_4d,
        COUNT(*) FILTER (
          WHERE s.SearchDate > t.timestamp - INTERVAL '14 days'
        )::DOUBLE AS n_searches_14d
      FROM target_keys t
      LEFT JOIN SearchInfo s ON s.UserID = t.UserID AND s.SearchDate < t.timestamp
      GROUP BY 1, 2
    ),
    -- historical 4-day windows ending at past search days: P(clicks>1)
    hist AS (
      SELECT t.timestamp, t.UserID,
        AVG(CASE WHEN cnt > 1 THEN 1.0 ELSE 0.0 END) AS hist_multiclick_rate
      FROM target_keys t
      JOIN (
        SELECT UserID, date_trunc('day', dt) AS day, COUNT(*) AS cnt
        FROM clicks
        GROUP BY 1, 2
      ) d ON d.UserID = t.UserID AND d.day < t.timestamp
        AND d.day >= t.timestamp - INTERVAL '28 days'
      GROUP BY 1, 2
    )
    SELECT
      b.timestamp, b.UserID,
      b.days_since_last_click, b.n_clicks_4d, b.n_clicks_14d, b.n_clicks_28d,
      s.n_searches_4d,
      CASE WHEN s.n_searches_14d > 0 THEN b.n_clicks_14d / s.n_searches_14d ELSE NULL END
        AS click_per_search_14d,
      h.hist_multiclick_rate
    FROM base b
    LEFT JOIN searches s USING (timestamp, UserID)
    LEFT JOIN hist h USING (timestamp, UserID)
    """
    return _run_feature_sql("rel-avito", keys, entity, sql, cols)


def _avito_user_visits(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "UserID"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_visit",
        "n_visits_4d",
        "n_distinct_ads_4d",
        "n_distinct_ads_14d",
        "n_visits_28d",
        "hist_multivisit_rate",
    ]
    sql = """
    WITH base AS (
      SELECT
        t.timestamp, t.UserID,
        date_diff('day', MAX(v.ViewDate), t.timestamp) AS days_since_last_visit,
        COUNT(*) FILTER (
          WHERE v.ViewDate > t.timestamp - INTERVAL '4 days'
        )::DOUBLE AS n_visits_4d,
        COUNT(DISTINCT v.AdID) FILTER (
          WHERE v.ViewDate > t.timestamp - INTERVAL '4 days'
        )::DOUBLE AS n_distinct_ads_4d,
        COUNT(DISTINCT v.AdID) FILTER (
          WHERE v.ViewDate > t.timestamp - INTERVAL '14 days'
        )::DOUBLE AS n_distinct_ads_14d,
        COUNT(*) FILTER (
          WHERE v.ViewDate > t.timestamp - INTERVAL '28 days'
        )::DOUBLE AS n_visits_28d
      FROM target_keys t
      LEFT JOIN VisitStream v ON v.UserID = t.UserID AND v.ViewDate < t.timestamp
      GROUP BY 1, 2
    ),
    hist AS (
      SELECT t.timestamp, t.UserID,
        AVG(CASE WHEN n_ads > 1 THEN 1.0 ELSE 0.0 END) AS hist_multivisit_rate
      FROM target_keys t
      JOIN (
        SELECT UserID, date_trunc('day', ViewDate) AS day,
               COUNT(DISTINCT AdID) AS n_ads
        FROM VisitStream
        GROUP BY 1, 2
      ) d ON d.UserID = t.UserID AND d.day < t.timestamp
        AND d.day >= t.timestamp - INTERVAL '28 days'
      GROUP BY 1, 2
    )
    SELECT b.*, h.hist_multivisit_rate
    FROM base b
    LEFT JOIN hist h USING (timestamp, UserID)
    """
    return _run_feature_sql("rel-avito", keys, entity, sql, cols)


def _event_user_ignore(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "user"
    keys, _ = _align_keys(X, time_col, entity)
    # Map task col `user` -> attendees.user_id
    cols = [
        "n_invited_7d",
        "n_invited_14d",
        "n_invited_30d",
        "n_yes_30d",
        "invite_share_30d",
        "hist_ignore_rate",
        "days_since_last_invite",
    ]
    sql = """
    WITH att AS (
      SELECT user_id AS user, start_time, status FROM event_attendees
    ),
    base AS (
      SELECT
        t.timestamp, t.user,
        COUNT(*) FILTER (
          WHERE a.status = 'invited' AND a.start_time > t.timestamp - INTERVAL '7 days'
        )::DOUBLE AS n_invited_7d,
        COUNT(*) FILTER (
          WHERE a.status = 'invited' AND a.start_time > t.timestamp - INTERVAL '14 days'
        )::DOUBLE AS n_invited_14d,
        COUNT(*) FILTER (
          WHERE a.status = 'invited' AND a.start_time > t.timestamp - INTERVAL '30 days'
        )::DOUBLE AS n_invited_30d,
        COUNT(*) FILTER (
          WHERE a.status IN ('yes','maybe') AND a.start_time > t.timestamp - INTERVAL '30 days'
        )::DOUBLE AS n_yes_30d,
        date_diff('day',
          MAX(a.start_time) FILTER (WHERE a.status = 'invited'),
          t.timestamp
        ) AS days_since_last_invite
      FROM target_keys t
      LEFT JOIN att a ON a.user = t.user AND a.start_time < t.timestamp
      GROUP BY 1, 2
    ),
    hist AS (
      SELECT t.timestamp, t.user,
        AVG(CASE WHEN n_inv > 2 THEN 1.0 ELSE 0.0 END) AS hist_ignore_rate
      FROM target_keys t
      JOIN (
        -- Filter start_time < t *before* week bucketing so partial weeks
        -- cannot count invites on/after the prediction timestamp.
        SELECT t2.timestamp, t2.user,
               date_trunc('week', a.start_time) AS wk,
               COUNT(*) FILTER (WHERE a.status = 'invited') AS n_inv
        FROM target_keys t2
        JOIN att a ON a.user = t2.user AND a.start_time < t2.timestamp
        GROUP BY 1, 2, 3
      ) w ON w.timestamp = t.timestamp AND w.user = t.user
        AND w.wk < t.timestamp
        AND w.wk >= t.timestamp - INTERVAL '90 days'
      GROUP BY 1, 2
    )
    SELECT
      b.*,
      CASE WHEN (b.n_invited_30d + b.n_yes_30d) > 0
        THEN b.n_invited_30d / (b.n_invited_30d + b.n_yes_30d) ELSE NULL END
        AS invite_share_30d,
      h.hist_ignore_rate
    FROM base b
    LEFT JOIN hist h USING (timestamp, user)
    """
    return _run_feature_sql("rel-event", keys, entity, sql, cols)


def _event_user_repeat(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "user"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_yes",
        "n_yes_14d",
        "n_yes_30d",
        "n_yes_90d",
        "hist_repeat_rate",
        "n_invited_14d",
    ]
    sql = """
    WITH att AS (
      SELECT user_id AS user, start_time, status FROM event_attendees
    ),
    base AS (
      SELECT
        t.timestamp, t.user,
        date_diff('day',
          MAX(a.start_time) FILTER (WHERE a.status IN ('yes','maybe')),
          t.timestamp
        ) AS days_since_last_yes,
        COUNT(*) FILTER (
          WHERE a.status IN ('yes','maybe')
            AND a.start_time > t.timestamp - INTERVAL '14 days'
        )::DOUBLE AS n_yes_14d,
        COUNT(*) FILTER (
          WHERE a.status IN ('yes','maybe')
            AND a.start_time > t.timestamp - INTERVAL '30 days'
        )::DOUBLE AS n_yes_30d,
        COUNT(*) FILTER (
          WHERE a.status IN ('yes','maybe')
            AND a.start_time > t.timestamp - INTERVAL '90 days'
        )::DOUBLE AS n_yes_90d,
        COUNT(*) FILTER (
          WHERE a.status = 'invited'
            AND a.start_time > t.timestamp - INTERVAL '14 days'
        )::DOUBLE AS n_invited_14d
      FROM target_keys t
      LEFT JOIN att a ON a.user = t.user AND a.start_time < t.timestamp
      GROUP BY 1, 2
    ),
    hist AS (
      SELECT t.timestamp, t.user,
        AVG(CASE WHEN n_yes > 0 THEN 1.0 ELSE 0.0 END) AS hist_repeat_rate
      FROM target_keys t
      JOIN (
        -- Filter start_time < t *before* week bucketing (same as hist_ignore_rate).
        SELECT t2.timestamp, t2.user,
               date_trunc('week', a.start_time) AS wk,
               COUNT(*) FILTER (WHERE a.status IN ('yes','maybe')) AS n_yes
        FROM target_keys t2
        JOIN att a ON a.user = t2.user AND a.start_time < t2.timestamp
        GROUP BY 1, 2, 3
      ) w ON w.timestamp = t.timestamp AND w.user = t.user
        AND w.wk < t.timestamp
        AND w.wk >= t.timestamp - INTERVAL '90 days'
      GROUP BY 1, 2
    )
    SELECT b.*, h.hist_repeat_rate
    FROM base b
    LEFT JOIN hist h USING (timestamp, user)
    """
    return _run_feature_sql("rel-event", keys, entity, sql, cols)


def _f1_driver_dnf(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "driverId"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_race",
        "dnf_rate_career",
        "dnf_rate_last5",
        "dnf_rate_last10",
        "n_races_365d",
        "constructor_dnf_rate",
    ]
    sql = """
    WITH res AS (
      SELECT r.*, CASE WHEN r.statusId != 1 THEN 1.0 ELSE 0.0 END AS is_dnf
      FROM results r
    ),
    base AS (
      SELECT
        t.timestamp, t.driverId,
        date_diff('day', MAX(r.date), t.timestamp) AS days_since_last_race,
        AVG(r.is_dnf) AS dnf_rate_career,
        COUNT(*) FILTER (WHERE r.date > t.timestamp - INTERVAL '365 days')::DOUBLE AS n_races_365d
      FROM target_keys t
      LEFT JOIN res r ON r.driverId = t.driverId AND r.date < t.timestamp
      GROUP BY 1, 2
    ),
    ranked AS (
      SELECT t.timestamp, t.driverId, r.is_dnf, r.constructorId, r.date,
        ROW_NUMBER() OVER (
          PARTITION BY t.timestamp, t.driverId ORDER BY r.date DESC
        ) AS rn
      FROM target_keys t
      JOIN res r ON r.driverId = t.driverId AND r.date < t.timestamp
    ),
    last5 AS (
      SELECT timestamp, driverId, AVG(is_dnf) AS dnf_rate_last5
      FROM ranked WHERE rn <= 5 GROUP BY 1, 2
    ),
    last10 AS (
      SELECT timestamp, driverId, AVG(is_dnf) AS dnf_rate_last10
      FROM ranked WHERE rn <= 10 GROUP BY 1, 2
    ),
    last_ctor AS (
      SELECT timestamp, driverId, constructorId FROM ranked WHERE rn = 1
    ),
    ctor AS (
      SELECT t.timestamp, t.driverId, AVG(r.is_dnf) AS constructor_dnf_rate
      FROM target_keys t
      JOIN last_ctor lc USING (timestamp, driverId)
      JOIN res r ON r.constructorId = lc.constructorId AND r.date < t.timestamp
        AND r.date > t.timestamp - INTERVAL '365 days'
      GROUP BY 1, 2
    )
    SELECT
      b.timestamp, b.driverId,
      b.days_since_last_race,
      b.dnf_rate_career,
      l5.dnf_rate_last5,
      l10.dnf_rate_last10,
      b.n_races_365d,
      c.constructor_dnf_rate
    FROM base b
    LEFT JOIN last5 l5 USING (timestamp, driverId)
    LEFT JOIN last10 l10 USING (timestamp, driverId)
    LEFT JOIN ctor c USING (timestamp, driverId)
    """
    return _run_feature_sql("rel-f1", keys, entity, sql, cols)


def _f1_driver_top3(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "driverId"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_quali",
        "top3_rate_career",
        "top3_rate_last5",
        "mean_quali_pos_last5",
        "best_quali_pos_365d",
        "constructor_top3_rate",
    ]
    sql = """
    WITH q AS (
      SELECT *, CASE WHEN position <= 3 THEN 1.0 ELSE 0.0 END AS is_top3
      FROM qualifying
    ),
    base AS (
      SELECT
        t.timestamp, t.driverId,
        date_diff('day', MAX(q.date), t.timestamp) AS days_since_last_quali,
        AVG(q.is_top3) AS top3_rate_career,
        MIN(q.position) FILTER (
          WHERE q.date > t.timestamp - INTERVAL '365 days'
        ) AS best_quali_pos_365d
      FROM target_keys t
      LEFT JOIN q ON q.driverId = t.driverId AND q.date < t.timestamp
      GROUP BY 1, 2
    ),
    ranked AS (
      SELECT t.timestamp, t.driverId, q.is_top3, q.position, q.constructorId,
        ROW_NUMBER() OVER (
          PARTITION BY t.timestamp, t.driverId ORDER BY q.date DESC
        ) AS rn
      FROM target_keys t
      JOIN q ON q.driverId = t.driverId AND q.date < t.timestamp
    ),
    last5 AS (
      SELECT timestamp, driverId,
        AVG(is_top3) AS top3_rate_last5,
        AVG(position) AS mean_quali_pos_last5
      FROM ranked WHERE rn <= 5 GROUP BY 1, 2
    ),
    last_ctor AS (
      SELECT timestamp, driverId, constructorId FROM ranked WHERE rn = 1
    ),
    ctor AS (
      SELECT t.timestamp, t.driverId, AVG(q.is_top3) AS constructor_top3_rate
      FROM target_keys t
      JOIN last_ctor lc USING (timestamp, driverId)
      JOIN q ON q.constructorId = lc.constructorId AND q.date < t.timestamp
        AND q.date > t.timestamp - INTERVAL '365 days'
      GROUP BY 1, 2
    )
    SELECT
      b.timestamp, b.driverId,
      b.days_since_last_quali,
      b.top3_rate_career,
      l5.top3_rate_last5,
      l5.mean_quali_pos_last5,
      b.best_quali_pos_365d,
      c.constructor_top3_rate
    FROM base b
    LEFT JOIN last5 l5 USING (timestamp, driverId)
    LEFT JOIN ctor c USING (timestamp, driverId)
    """
    return _run_feature_sql("rel-f1", keys, entity, sql, cols)





PERIOD_RULE_SPECS: Dict[str, Dict[str, Any]] = {
  "relbench::rel-amazon::user-churn": {
    "dataset": "rel-amazon",
    "entity_col": "customer_id",
    "activity_table": "review",
    "fk_col": "customer_id",
    "time_col_act": "review_time",
    "numeric_cols": [
      "rating",
      "verified"
    ]
  },
  "relbench::rel-amazon::item-churn": {
    "dataset": "rel-amazon",
    "entity_col": "product_id",
    "activity_table": "review",
    "fk_col": "product_id",
    "time_col_act": "review_time",
    "numeric_cols": [
      "rating",
      "verified"
    ]
  },
  "relbench::rel-avito::user-clicks": {
    "dataset": "rel-avito",
    "entity_col": "UserID",
    "activity_table": "VisitStream",
    "fk_col": "UserID",
    "time_col_act": "ViewDate",
    "numeric_cols": [
      "IPID"
    ]
  },
  "relbench::rel-avito::user-visits": {
    "dataset": "rel-avito",
    "entity_col": "UserID",
    "activity_table": "VisitStream",
    "fk_col": "UserID",
    "time_col_act": "ViewDate",
    "numeric_cols": [
      "IPID"
    ]
  },
  "relbench::rel-event::user-ignore": {
    "dataset": "rel-event",
    "entity_col": "user",
    "activity_table": "event_interest",
    "fk_col": "user",
    "time_col_act": "timestamp",
    "numeric_cols": [
      "invited",
      "interested",
      "not_interested"
    ]
  },
  "relbench::rel-event::user-repeat": {
    "dataset": "rel-event",
    "entity_col": "user",
    "activity_table": "event_interest",
    "fk_col": "user",
    "time_col_act": "timestamp",
    "numeric_cols": [
      "invited",
      "interested",
      "not_interested"
    ]
  },
  "relbench::rel-f1::driver-dnf": {
    "dataset": "rel-f1",
    "entity_col": "driverId",
    "activity_table": "standings",
    "fk_col": "driverId",
    "time_col_act": "date",
    "numeric_cols": [
      "points",
      "position",
      "wins"
    ]
  },
  "relbench::rel-f1::driver-top3": {
    "dataset": "rel-f1",
    "entity_col": "driverId",
    "activity_table": "standings",
    "fk_col": "driverId",
    "time_col_act": "date",
    "numeric_cols": [
      "points",
      "position",
      "wins"
    ]
  },
  "relbench::rel-hm::user-churn": {
    "dataset": "rel-hm",
    "entity_col": "customer_id",
    "activity_table": "transactions",
    "fk_col": "customer_id",
    "time_col_act": "t_dat",
    "numeric_cols": [
      "price",
      "sales_channel_id"
    ]
  },
  "relbench::rel-stack::user-badge": {
    "dataset": "rel-stack",
    "entity_col": "UserId",
    "activity_table": "badges",
    "fk_col": "UserId",
    "time_col_act": "Date",
    "numeric_cols": [
      "Class",
      "TagBased"
    ]
  },
  "relbench::rel-stack::user-engagement": {
    "dataset": "rel-stack",
    "entity_col": "OwnerUserId",
    "activity_table": "posts",
    "fk_col": "OwnerUserId",
    "time_col_act": "CreationDate",
    "numeric_cols": [
      "PostTypeId"
    ]
  },
  "relbench::rel-trial::study-outcome": {
    "dataset": "rel-trial",
    "entity_col": "nct_id",
    "activity_table": "outcome_analyses",
    "fk_col": "nct_id",
    "time_col_act": "date",
    "numeric_cols": [
      "param_value",
      "dispersion_value",
      "p_value",
      "ci_percent",
      "ci_lower_limit",
      "ci_upper_limit"
    ]
  },
  "relbench::rel-trial::study-adverse": {
    "dataset": "rel-trial",
    "entity_col": "nct_id",
    "activity_table": "reported_event_totals",
    "fk_col": "nct_id",
    "time_col_act": "date",
    "numeric_cols": [
      "subjects_affected",
      "subjects_at_risk"
    ]
  },
  # --- regression tasks (same activity schemas as sibling clf tasks where applicable) ---
  "relbench::rel-amazon::item-ltv": {
    "dataset": "rel-amazon",
    "entity_col": "product_id",
    "activity_table": "review",
    "fk_col": "product_id",
    "time_col_act": "review_time",
    "numeric_cols": [
      "rating",
      "verified"
    ]
  },
  "relbench::rel-amazon::user-ltv": {
    "dataset": "rel-amazon",
    "entity_col": "customer_id",
    "activity_table": "review",
    "fk_col": "customer_id",
    "time_col_act": "review_time",
    "numeric_cols": [
      "rating",
      "verified"
    ]
  },
  "relbench::rel-avito::ad-ctr": {
    "dataset": "rel-avito",
    "entity_col": "AdID",
    "activity_table": "SearchStream",
    "fk_col": "AdID",
    "time_col_act": "SearchDate",
    "numeric_cols": [
      "Position",
      "HistCTR",
      "IsClick"
    ]
  },
  "relbench::rel-event::user-attendance": {
    "dataset": "rel-event",
    "entity_col": "user",
    "activity_table": "event_interest",
    "fk_col": "user",
    "time_col_act": "timestamp",
    "numeric_cols": [
      "invited",
      "interested",
      "not_interested"
    ]
  },
  "relbench::rel-f1::driver-position": {
    "dataset": "rel-f1",
    "entity_col": "driverId",
    "activity_table": "standings",
    "fk_col": "driverId",
    "time_col_act": "date",
    "numeric_cols": [
      "points",
      "position",
      "wins"
    ]
  },
  "relbench::rel-hm::item-sales": {
    "dataset": "rel-hm",
    "entity_col": "article_id",
    "activity_table": "transactions",
    "fk_col": "article_id",
    "time_col_act": "t_dat",
    "numeric_cols": [
      "price",
      "sales_channel_id"
    ]
  },
  "relbench::rel-stack::post-votes": {
    "dataset": "rel-stack",
    "entity_col": "PostId",
    "activity_table": "votes",
    "fk_col": "PostId",
    "time_col_act": "CreationDate",
    "numeric_cols": [
      "VoteTypeId"
    ]
  },
  "relbench::rel-trial::site-success": {
    "dataset": "rel-trial",
    "entity_col": "facility_id",
    "activity_table": "facilities_studies",
    "fk_col": "facility_id",
    "time_col_act": "date",
    "numeric_cols": []
  }
}


def _qident(name: str) -> str:
    """Quote DuckDB identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def _period_dfs_rule_from_spec(
    X: pd.DataFrame,
    time_col: str,
    spec: Dict[str, Any],
    periods: tuple[str, ...] = ("week", "month", "quarter", "year"),
) -> pd.DataFrame:
    """Generic closest-activity period DFS-agg rule.

    Per period: bucket activity by date_trunc(period, time), drop current bucket,
    compute COUNT + MAX/MIN/MEAN/STD/MEDIAN on numeric cols, then outer
    AVG/MAX/MIN/STD/MEDIAN over buckets.
    """
    entity = spec["entity_col"]
    keys, _ = _align_keys(X, time_col, entity)
    table = spec["activity_table"]
    fk = spec["fk_col"]
    tcol = spec["time_col_act"]
    num_cols = list(spec.get("numeric_cols") or [])
    dataset = spec["dataset"]

    # per-bucket stats: count + 5 aggs per numeric col
    bucket_stats = ["count"]
    bucket_select = [f"COUNT(*)::DOUBLE AS count"]
    for c in num_cols:
        qc = _qident(c)
        expr = f"TRY_CAST({qc} AS DOUBLE)"
        for agg, alias in [
            ("MAX", f"max_{c}"),
            ("MIN", f"min_{c}"),
            ("AVG", f"mean_{c}"),
            ("stddev_samp", f"std_{c}"),
            ("median", f"median_{c}"),
        ]:
            bucket_stats.append(alias)
            bucket_select.append(f"{agg}({expr}) AS {_qident(alias)}")

    outers = ["avg", "max", "min", "std", "median"]
    allowed = {"week", "month", "quarter", "year", "day"}
    periods = tuple(p.lower() for p in periods)
    for p in periods:
        if p not in allowed:
            raise ValueError(f"unsupported period {p!r}")

    frames = []
    all_cols: List[str] = []
    qt, qfk, qtcol = _qident(table), _qident(fk), _qident(tcol)
    qentity = _qident(entity)

    for period in periods:
        cols = [f"{o}_{period}_{s}" for s in bucket_stats for o in outers]
        all_cols.extend(cols)
        outer_parts = []
        for s in bucket_stats:
            qs = _qident(s)
            outer_parts.append(f"AVG(b.{qs}) AS {_qident(f'avg_{period}_{s}')}")
            outer_parts.append(f"MAX(b.{qs}) AS {_qident(f'max_{period}_{s}')}")
            outer_parts.append(f"MIN(b.{qs}) AS {_qident(f'min_{period}_{s}')}")
            outer_parts.append(f"stddev_samp(b.{qs}) AS {_qident(f'std_{period}_{s}')}")
            outer_parts.append(f"median(b.{qs}) AS {_qident(f'median_{period}_{s}')}")
        outer_sql = ",\n        ".join(outer_parts)
        bucket_sql = ",\n            ".join(bucket_select)
        sql = f"""
        WITH buckets AS (
          SELECT
            {qfk} AS entity_id,
            date_trunc('{period}', {qtcol}) AS bucket,
            {bucket_sql}
          FROM {qt}
          GROUP BY 1, 2
        ),
        hist AS (
          SELECT
            t.timestamp, t.{qentity},
            {outer_sql}
          FROM target_keys t
          LEFT JOIN buckets b
            ON b.entity_id = t.{qentity}
           AND b.bucket < date_trunc('{period}', t.timestamp)
          GROUP BY 1, 2
        )
        SELECT * FROM hist
        """
        feat = _run_feature_sql(dataset, keys, entity, sql, cols)
        frames.append(feat)

    out = frames[0]
    for f in frames[1:]:
        key_cols = [c for c in f.columns if c in ("timestamp", entity)]
        out = pd.concat([out, f.drop(columns=key_cols, errors="ignore")], axis=1)
    feat_cols = [c for c in all_cols if c in out.columns]
    return out[feat_cols]


def _multi_period_dfs_rule_for_task(task_id: str, X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if task_id not in PERIOD_RULE_SPECS:
        raise KeyError(f"No PERIOD_RULE_SPECS for {task_id}")
    return _period_dfs_rule_from_spec(X, time_col, PERIOD_RULE_SPECS[task_id])



def _hm_period_dfs_rule(
    X: pd.DataFrame,
    time_col: str,
    periods: tuple[str, ...] = ("week", "month", "quarter", "year"),
) -> pd.DataFrame:
    return _period_dfs_rule_from_spec(
        X, time_col, PERIOD_RULE_SPECS["relbench::rel-hm::user-churn"], periods=periods
    )


def _hm_weekly_dfs_rule(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    return _hm_period_dfs_rule(X, time_col, periods=("week",))


def _hm_multi_period_dfs_rule(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    return _hm_period_dfs_rule(X, time_col, periods=("week", "month", "quarter", "year"))



def _stack_user_badge(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "UserId"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_badge",
        "n_badges_91d",
        "n_badges_365d",
        "n_badges_lifetime",
        "n_posts_91d",
        "n_votes_91d",
        "n_comments_91d",
        "hist_badge_rate",
    ]
    sql = """
    WITH base AS (
      SELECT
        t.timestamp, t.UserId,
        date_diff('day', MAX(b.Date), t.timestamp) AS days_since_last_badge,
        COUNT(*) FILTER (
          WHERE b.Date > t.timestamp - INTERVAL '91 days'
        )::DOUBLE AS n_badges_91d,
        COUNT(*) FILTER (
          WHERE b.Date > t.timestamp - INTERVAL '365 days'
        )::DOUBLE AS n_badges_365d,
        COUNT(*)::DOUBLE AS n_badges_lifetime
      FROM target_keys t
      LEFT JOIN badges b ON b.UserId = t.UserId AND b.Date < t.timestamp
      GROUP BY 1, 2
    ),
    posts_f AS (
      SELECT t.timestamp, t.UserId, COUNT(*)::DOUBLE AS n_posts_91d
      FROM target_keys t
      JOIN posts p ON p.OwnerUserId = t.UserId
        AND p.CreationDate < t.timestamp
        AND p.CreationDate > t.timestamp - INTERVAL '91 days'
      GROUP BY 1, 2
    ),
    votes_f AS (
      SELECT t.timestamp, t.UserId, COUNT(*)::DOUBLE AS n_votes_91d
      FROM target_keys t
      JOIN votes v ON v.UserId = t.UserId
        AND v.CreationDate < t.timestamp
        AND v.CreationDate > t.timestamp - INTERVAL '91 days'
      GROUP BY 1, 2
    ),
    comments_f AS (
      SELECT t.timestamp, t.UserId, COUNT(*)::DOUBLE AS n_comments_91d
      FROM target_keys t
      JOIN comments c ON c.UserId = t.UserId
        AND c.CreationDate < t.timestamp
        AND c.CreationDate > t.timestamp - INTERVAL '91 days'
      GROUP BY 1, 2
    ),
    hist AS (
      SELECT t.timestamp, t.UserId,
        AVG(CASE WHEN n_b > 0 THEN 1.0 ELSE 0.0 END) AS hist_badge_rate
      FROM target_keys t
      JOIN (
        SELECT UserId, date_trunc('quarter', Date) AS qtr, COUNT(*) AS n_b
        FROM badges GROUP BY 1, 2
      ) q ON q.UserId = t.UserId AND q.qtr < t.timestamp
        AND q.qtr >= t.timestamp - INTERVAL '730 days'
      GROUP BY 1, 2
    )
    SELECT
      b.*,
      COALESCE(p.n_posts_91d, 0) AS n_posts_91d,
      COALESCE(v.n_votes_91d, 0) AS n_votes_91d,
      COALESCE(c.n_comments_91d, 0) AS n_comments_91d,
      h.hist_badge_rate
    FROM base b
    LEFT JOIN posts_f p USING (timestamp, UserId)
    LEFT JOIN votes_f v USING (timestamp, UserId)
    LEFT JOIN comments_f c USING (timestamp, UserId)
    LEFT JOIN hist h USING (timestamp, UserId)
    """
    return _run_feature_sql("rel-stack", keys, entity, sql, cols)


def _stack_user_engagement(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "OwnerUserId"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_engagement",
        "n_posts_91d",
        "n_votes_91d",
        "n_comments_91d",
        "n_eng_91d",
        "n_eng_365d",
        "n_badges_91d",
        "hist_engage_rate",
    ]
    sql = """
    WITH eng AS (
      SELECT OwnerUserId AS uid, CreationDate AS dt, 'post' AS kind FROM posts
      WHERE OwnerUserId IS NOT NULL AND OwnerUserId != -1
      UNION ALL
      SELECT UserId, CreationDate, 'vote' FROM votes
      WHERE UserId IS NOT NULL AND UserId != -1
      UNION ALL
      SELECT UserId, CreationDate, 'comment' FROM comments
      WHERE UserId IS NOT NULL AND UserId != -1
    ),
    base AS (
      SELECT
        t.timestamp, t.OwnerUserId,
        date_diff('day', MAX(e.dt), t.timestamp) AS days_since_last_engagement,
        COUNT(*) FILTER (
          WHERE e.kind = 'post' AND e.dt > t.timestamp - INTERVAL '91 days'
        )::DOUBLE AS n_posts_91d,
        COUNT(*) FILTER (
          WHERE e.kind = 'vote' AND e.dt > t.timestamp - INTERVAL '91 days'
        )::DOUBLE AS n_votes_91d,
        COUNT(*) FILTER (
          WHERE e.kind = 'comment' AND e.dt > t.timestamp - INTERVAL '91 days'
        )::DOUBLE AS n_comments_91d,
        COUNT(*) FILTER (
          WHERE e.dt > t.timestamp - INTERVAL '91 days'
        )::DOUBLE AS n_eng_91d,
        COUNT(*) FILTER (
          WHERE e.dt > t.timestamp - INTERVAL '365 days'
        )::DOUBLE AS n_eng_365d
      FROM target_keys t
      LEFT JOIN eng e ON e.uid = t.OwnerUserId AND e.dt < t.timestamp
      GROUP BY 1, 2
    ),
    badges_f AS (
      SELECT t.timestamp, t.OwnerUserId,
        COUNT(*) FILTER (
          WHERE b.Date > t.timestamp - INTERVAL '91 days'
        )::DOUBLE AS n_badges_91d
      FROM target_keys t
      LEFT JOIN badges b ON b.UserId = t.OwnerUserId AND b.Date < t.timestamp
      GROUP BY 1, 2
    ),
    hist AS (
      SELECT t.timestamp, t.OwnerUserId,
        AVG(CASE WHEN n_e > 0 THEN 1.0 ELSE 0.0 END) AS hist_engage_rate
      FROM target_keys t
      JOIN (
        SELECT uid, date_trunc('quarter', dt) AS qtr, COUNT(*) AS n_e
        FROM eng GROUP BY 1, 2
      ) q ON q.uid = t.OwnerUserId AND q.qtr < t.timestamp
        AND q.qtr >= t.timestamp - INTERVAL '730 days'
      GROUP BY 1, 2
    )
    SELECT
      b.*, bf.n_badges_91d, h.hist_engage_rate
    FROM base b
    LEFT JOIN badges_f bf USING (timestamp, OwnerUserId)
    LEFT JOIN hist h USING (timestamp, OwnerUserId)
    """
    return _run_feature_sql("rel-stack", keys, entity, sql, cols)


# Keep study-outcome path via existing module
def _study_outcome(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    from .study_outcome_hist_features import StudyOutcomeHistFeatureBuilder

    builder = StudyOutcomeHistFeatureBuilder()
    return builder.compute(X, rdb=None, time_col=time_col)



def _hm_user_churn(X: pd.DataFrame, time_col: str) -> pd.DataFrame:
    entity = "customer_id"
    keys, _ = _align_keys(X, time_col, entity)
    cols = [
        "days_since_last_purchase",
        "n_purchases_7d",
        "n_purchases_30d",
        "n_purchases_90d",
        "n_distinct_articles_30d",
        "sum_price_30d",
        "hist_weekly_churn_rate",
    ]
    sql = """
    WITH base AS (
      SELECT
        t.timestamp, t.customer_id,
        date_diff('day', MAX(tr.t_dat), t.timestamp) AS days_since_last_purchase,
        COUNT(*) FILTER (
          WHERE tr.t_dat > t.timestamp - INTERVAL '7 days'
        )::DOUBLE AS n_purchases_7d,
        COUNT(*) FILTER (
          WHERE tr.t_dat > t.timestamp - INTERVAL '30 days'
        )::DOUBLE AS n_purchases_30d,
        COUNT(*) FILTER (
          WHERE tr.t_dat > t.timestamp - INTERVAL '90 days'
        )::DOUBLE AS n_purchases_90d,
        COUNT(DISTINCT tr.article_id) FILTER (
          WHERE tr.t_dat > t.timestamp - INTERVAL '30 days'
        )::DOUBLE AS n_distinct_articles_30d,
        SUM(tr.price) FILTER (
          WHERE tr.t_dat > t.timestamp - INTERVAL '30 days'
        ) AS sum_price_30d
      FROM target_keys t
      LEFT JOIN transactions tr
        ON tr.customer_id = t.customer_id AND tr.t_dat < t.timestamp
      GROUP BY 1, 2
    ),
    -- weekly activity: weeks with 0 purchases after an active week ≈ churn proxy
    weekly AS (
      SELECT customer_id, date_trunc('week', t_dat) AS wk, COUNT(*) AS n
      FROM transactions
      GROUP BY 1, 2
    ),
    hist AS (
      SELECT t.timestamp, t.customer_id,
        AVG(CASE WHEN w2.n IS NULL THEN 1.0 ELSE 0.0 END) AS hist_weekly_churn_rate
      FROM target_keys t
      JOIN weekly w ON w.customer_id = t.customer_id
        AND w.wk < t.timestamp - INTERVAL '7 days'
        AND w.wk >= t.timestamp - INTERVAL '90 days'
      LEFT JOIN weekly w2 ON w2.customer_id = w.customer_id
        AND w2.wk = w.wk + INTERVAL '7 days'
      GROUP BY 1, 2
    )
    SELECT b.*, h.hist_weekly_churn_rate
    FROM base b
    LEFT JOIN hist h USING (timestamp, customer_id)
    """
    return _run_feature_sql("rel-hm", keys, entity, sql, cols)



REGISTRY: Dict[str, Callable[[pd.DataFrame, str], pd.DataFrame]] = {
    "relbench::rel-amazon::item-churn": _amazon_item_churn,
    "relbench::rel-amazon::user-churn": _amazon_user_churn,
    "relbench::rel-avito::user-clicks": _avito_user_clicks,
    "relbench::rel-avito::user-visits": _avito_user_visits,
    "relbench::rel-event::user-ignore": _event_user_ignore,
    "relbench::rel-event::user-repeat": _event_user_repeat,
    "relbench::rel-f1::driver-dnf": _f1_driver_dnf,
    "relbench::rel-f1::driver-top3": _f1_driver_top3,
    "relbench::rel-hm::user-churn": _hm_user_churn,
    "relbench::rel-stack::user-badge": _stack_user_badge,
    "relbench::rel-stack::user-engagement": _stack_user_engagement,
    "relbench::rel-trial::study-outcome": _study_outcome,
}


def compute_sql_features_for_task(
    task_id: str,
    X: pd.DataFrame,
    time_col: str,
) -> pd.DataFrame:
    import os
    variant = os.environ.get("RDBLEARN_SQL_VARIANT", "").strip().lower()
    if variant in ("weekly_dfs_rule", "weekly_rule", "hm_weekly_dfs_rule") and task_id.endswith("rel-hm::user-churn"):
        feats = _hm_weekly_dfs_rule(X, time_col)
        return feats.rename(columns={c: f"sql__{c}" for c in feats.columns})
    if variant in (
        "multi_period_dfs_rule",
        "period_dfs_rule",
        "hm_multi_period_dfs_rule",
        "week_month_quarter_year",
    ):
        feats = _multi_period_dfs_rule_for_task(task_id, X, time_col)
        return feats.rename(columns={c: f"sql__{c}" for c in feats.columns})
    if variant.startswith("periods:") and task_id in PERIOD_RULE_SPECS:
        periods = tuple(p.strip() for p in variant.split(":", 1)[1].split(",") if p.strip())
        feats = _period_dfs_rule_from_spec(X, time_col, PERIOD_RULE_SPECS[task_id], periods=periods)
        return feats.rename(columns={c: f"sql__{c}" for c in feats.columns})
    if task_id not in REGISTRY:
        raise KeyError(f"No SQL feature builder for task_id={task_id}")
    feats = REGISTRY[task_id](X, time_col)
    return feats.rename(columns={c: f"sql__{c}" for c in feats.columns})


def append_sql_features_for_task(
    dfs_features: pd.DataFrame,
    X: pd.DataFrame,
    task_id: str,
    cutoff_time_column: Optional[str] = "timestamp",
) -> pd.DataFrame:
    """Append task-specific SQL features to an in-memory DFS frame."""
    existing = [c for c in dfs_features.columns if c.startswith("sql__")]
    if existing:
        return dfs_features
    time_col = cutoff_time_column or "timestamp"
    # Prefer keys from X; fall back to columns present on dfs_features
    key_src = X
    extra = compute_sql_features_for_task(task_id, key_src, time_col)
    base = dfs_features.reset_index(drop=True)
    if len(base) != len(extra):
        raise RuntimeError(
            f"Cannot append SQL features: dfs={len(base)} sql={len(extra)} task={task_id}"
        )
    out = pd.concat([base, extra], axis=1)
    logger.info(
        "sql_extra_features[{}]: appended {} cols ({} -> {})",
        task_id,
        extra.shape[1],
        base.shape[1],
        out.shape[1],
    )
    return out
