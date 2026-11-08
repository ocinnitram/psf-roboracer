import sys
from pathlib import Path

SF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SF_DIR / "build"))
sys.path.insert(0, str(SF_DIR))
