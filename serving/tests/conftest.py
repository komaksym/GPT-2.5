from pathlib import Path
import sys

SERVING_DIR = Path(__file__).resolve().parents[1]

if str(SERVING_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_DIR))
