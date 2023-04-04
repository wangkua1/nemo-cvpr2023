"""
python -m VIBE.lib.data_utils.check_processed_data
"""

import os
import os.path as osp
import pickle as pkl
import joblib
import numpy as np
from lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR
import ipdb


dataset = joblib.load(osp.join(VIBE_DB_DIR, '3dpw_train_db.pt'))

ipdb.set_trace()