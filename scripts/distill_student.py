"""Train the paper's deployed student from a local source checkpoint."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from experiments.students.cli import main
    main()
