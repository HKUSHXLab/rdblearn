"""Utilities to transform TabPFN-style dataset metadata into per-task metadata files.

Expected original layout (example for dataset `rel-amazon`)::

	/root/autodl-tmp/tabpfn_data/rel-amazon/metadata.yaml

This script will, for each task entry in that metadata file, create a
task-specific directory and a new `metadata.yaml`, e.g.::

	/root/autodl-tmp/tabpfn_data/rel-amazon/item-ltv/metadata.yaml

The transform logic per user requirements:

* Identify the primary key column in the **task's columns** list
  (the column whose ``dtype`` is ``primary_key``).
* The target table is given by ``task["target_table"]``.
* Look up that table in the top-level ``tables`` section and find its
  primary key column (``dtype == "primary_key"``).
* Build a key mapping of the form::

	  {<task_primary_key_name>: "<target_table>.<table_primary_key_name>"}

* Keep ``target_column``, ``time_column`` and ``task_type`` from the
  original task definition.

The generated per-task metadata file is intentionally minimal and can be
extended later if needed.

python old_scripts/process_metadata.py --dataset_name rel-amazon
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
	with path.open("r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def _dump_yaml(data: Dict[str, Any], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		yaml.safe_dump(data, f, sort_keys=False)


def _find_table_primary_key(tables: List[Dict[str, Any]], table_name: str) -> Optional[str]:
	"""Return the primary key column name for a given table, or None.

	``tables`` is the list from the original metadata["tables"].
	"""

	for table in tables:
		if table.get("name") != table_name:
			continue
		for col in table.get("columns", []):
			if col.get("dtype") == "primary_key":
				return col.get("name")
	return None


def _find_task_primary_key(task: Dict[str, Any]) -> Optional[str]:
	"""Return the primary key column name from a task definition, or None."""

	for col in task.get("columns", []):
		if col.get("dtype") == "primary_key":
			return col.get("name")
	return None


def build_task_metadata(original_meta: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
	"""Construct a minimal per-task metadata dictionary.

	The schema is small by design; it captures the key mapping and core
	task attributes that downstream code needs.
	"""

	tables = original_meta.get("tables", [])

	task_name = task.get("name")
	target_table = task.get("target_table")
	if target_table is None:
		raise ValueError(f"Task '{task_name}' has no target_table defined in metadata.")

	task_pk = _find_task_primary_key(task)
	if task_pk is None:
		raise ValueError(f"Task '{task_name}' has no primary_key column in its columns list.")

	table_pk = _find_table_primary_key(tables, target_table)
	if table_pk is None:
		raise ValueError(
			f"Target table '{target_table}' for task '{task_name}' has no primary_key column in tables section."
		)

	# Represent key mappings as a list of single-key dictionaries, e.g.
	# key_mappings:
	#   - item-ltv: product.item-ltv
	key_mappings = [{task_pk: f"{target_table}.{table_pk}"}]
	task_type = task.get("task_type")
	if task_type == "classification":
		task_type = "binary"

	task_metadata = {
		"task_name": task_name,
		"dataset_name": original_meta.get("datasetpy_name"),
		"target_table": target_table,
		"key_mappings": key_mappings,
		"target_column": task.get("target_column"),
		"time_column": task.get("time_column"),
		"task_type": task_type,
	}

	return task_metadata


def transform_dataset_metadata(dataset_dir: Path) -> None:
	"""Transform a dataset-level metadata.yaml into per-task metadata files.

	Parameters
	----------
	dataset_dir:
		Path to the dataset directory that contains ``metadata.yaml``.
	"""

	meta_path = dataset_dir / "metadata.yaml"
	if not meta_path.is_file():
		raise FileNotFoundError(f"metadata.yaml not found in {dataset_dir}")

	meta = _load_yaml(meta_path)

	tasks: List[Dict[str, Any]] = meta.get("tasks", [])
	if not tasks:
		raise ValueError(f"No tasks found in metadata.yaml at {meta_path}")

	for task in tasks:
		task_name = task.get("name")
		if not task_name:
			# Skip unnamed tasks to avoid bogus directories.
			continue

		per_task_meta = build_task_metadata(meta, task)

		out_dir = dataset_dir / task_name
		out_path = out_dir / "metadata.yaml"
		_dump_yaml(per_task_meta, out_path)

		print(f"Wrote per-task metadata for task '{task_name}' to {out_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Transform TabPFN metadata into per-task metadata files.")
	parser.add_argument(
		"dataset_dir",
		type=str,
		help="Path to the dataset directory containing metadata.yaml (e.g. /root/autodl-tmp/tabpfn_data/rel-amazon)",
	)
	return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--root_dir", default="/root/autodl-tmp/tabpfn_data")
    args = parser.parse_args()
    transform_dataset_metadata(Path(args.root_dir) / args.dataset_name)
