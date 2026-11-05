"""Run the KD-soft or confidence pseudo-labeling baseline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from experiments.students.cli import main
    main(baseline=True)
