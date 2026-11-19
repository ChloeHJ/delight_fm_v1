"""Portable source and run roots, with optional adjacent release data."""
import os
from pathlib import Path
SRC=Path(__file__).resolve().parent
TASK=SRC.parent
RELEASE_DATA=TASK.parent/'huggingface/dataset'
DEFAULT_RUN=RELEASE_DATA if (RELEASE_DATA/'data/input_manifest.json').is_file() else TASK/'runs'
RUN=Path(os.environ.get('DELIGHT_FM_RUN_DIR',os.environ.get('V12_RUN_DIR',DEFAULT_RUN))).resolve()
