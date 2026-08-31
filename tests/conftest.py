"""Make the image server and Tapestry package importable for direct tests."""

from pathlib import Path
from sys import path as sys_path

root = Path(__file__).resolve().parent.parent
project_root = root.parents[1]
sys_path.insert(0, str(project_root))
sys_path.insert(0, str(root))
sys_path.insert(0, str(root / "container"))
