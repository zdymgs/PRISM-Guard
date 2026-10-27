"""Run one CUHK-X outer fold with its own source checkpoint and targets."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from experiments.students.cli import main
    arguments = sys.argv[1:]
    if "--dataset" not in arguments:
        arguments = ["--dataset", "cuhkx", *arguments]
    main(arguments)
