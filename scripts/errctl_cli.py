"""errctl-cli -- launcher for the calibration layer as a command line tool.

Runs from a bare checkout with no installation:

    python scripts/errctl_cli.py demo    --out-dir out
    python scripts/errctl_cli.py advise  --demo --out-dir out
    python scripts/errctl_cli.py certify --csv cells.csv --label Cell_Label \
        --markers CD3 CD20 CD8 --alpha 0.10 --out-dir out
    python scripts/errctl_cli.py deploy  --source train.csv --target batch.csv \
        --label Cell_Label --k 500 --alpha 0.10 --out-dir out
    python scripts/errctl_cli.py advise  --source train.csv --target batch.csv \
        --label Cell_Label --k 500 --scan-k 250,500,1000,2000 --out-dir out
    python scripts/errctl_cli.py advise  --source train.csv \
        --target unlabelled_batch.csv --calibration-csv pilot_labels.csv \
        --label Cell_Label --calibration-id cell --k 500 --out-dir out

The implementation is `errctl.cli`; see its docstring for the file formats and
for what each run writes.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from errctl.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
