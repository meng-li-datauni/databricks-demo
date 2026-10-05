"""Runtime configuration shared by every step of the demo.

The same code runs in two places:

* on Databricks (job / notebook / serverless), where tables live in Unity Catalog
  (``catalog.schema.table``) and files live in a UC Volume (``/Volumes/...``);
* locally with open-source PySpark + Delta Lake, where tables live in a local
  Hive metastore (``schema.table``) and files live under ``./.local``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

LOCAL_ROOT = Path(os.environ.get("DEMO_LOCAL_ROOT", ".local")).resolve()
VOLUME_NAME = "demo"


def is_databricks() -> bool:
    """True when running on Databricks compute (cluster, serverless, pipeline)."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


@dataclass(frozen=True)
class DemoConfig:
    catalog: str | None
    schema: str

    @property
    def namespace(self) -> str:
        """Unity Catalog uses a three-level namespace; local Spark uses two."""
        return f"{self.catalog}.{self.schema}" if self.catalog else self.schema

    def table(self, name: str) -> str:
        return f"{self.namespace}.{name}"

    @property
    def volume_root(self) -> str:
        if is_databricks():
            return f"/Volumes/{self.catalog}/{self.schema}/{VOLUME_NAME}"
        return str(LOCAL_ROOT / "volume")

    @property
    def landing_dir(self) -> str:
        return f"{self.volume_root}/landing"

    @property
    def checkpoint_dir(self) -> str:
        return f"{self.volume_root}/_checkpoints"

    @property
    def schema_dir(self) -> str:
        return f"{self.volume_root}/_schemas"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> DemoConfig:
        return cls(catalog=args.catalog or None, schema=args.schema)


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    """CLI used by every entry point; job tasks pass ``--catalog``/``--schema``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--catalog",
        default=os.environ.get("DEMO_CATALOG", "main" if is_databricks() else None),
        help="Unity Catalog catalog (ignored locally)",
    )
    parser.add_argument(
        "--schema",
        default=os.environ.get("DEMO_SCHEMA", "databricks_demo"),
        help="Schema (database) holding the demo tables",
    )
    return parser
