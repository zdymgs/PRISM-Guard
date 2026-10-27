from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism.cli import build_main

if __name__ == "__main__":
    build_main()
