"""Single entry point: python run.py --experiment <config name in configs/>.

Everything else (model, dataset, seeds, stages, hyperparameters) lives in the config.
"""

import argparse

from morg.config import CONFIG_DIR, load_config
from morg.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True,
                        help="one of: " + ", ".join(sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))))
    run(load_config(parser.parse_args().experiment))
