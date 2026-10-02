"""Make the image server importable for direct tests."""

from pathlib import Path
from sys import path as sys_path

root = Path(__file__).resolve().parent.parent
sys_path.insert(0, str(root))
sys_path.insert(0, str(root / "container"))
