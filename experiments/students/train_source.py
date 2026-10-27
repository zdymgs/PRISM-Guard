"""Run a source initialization stage."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from experiments.students.source_cli import main
    main()
