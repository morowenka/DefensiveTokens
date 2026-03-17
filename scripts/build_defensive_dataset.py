"""Step 2: Build defensive training dataset.

50% clean samples, 50% with prompt injection in data field.
Target remains the original base_response.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import setup_script
from src.dataset import build_defensive_dataset

logger = logging.getLogger(__name__)


def main():
    config = setup_script("build_defensive")

    build_defensive_dataset(
        config["dataset"]["self_labeled_path"],
        config["dataset"]["defensive_path"],
        config["dataset"]["injection_ratio"],
    )


if __name__ == "__main__":
    main()
