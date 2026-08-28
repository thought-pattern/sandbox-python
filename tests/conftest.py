"""Make the submodule importable so tests can import manager and server directly."""

from pathlib import Path
from sys import path as sys_path

root = Path(__file__).resolve().parent.parent
sys_path.insert(0, str(root))
sys_path.insert(0, str(root / "container"))
