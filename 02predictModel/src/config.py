'''
超参数配置
'''
from pathlib import Path
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / '05Deploy'))
from runtime_paths import PROJECT_ROOT
_BASE = PROJECT_ROOT / '02predictModel'
RAW_DATA_DIR       = _BASE / 'data' / 'raw'
MODELS_DIR         = _BASE / 'models'
PROCESSED_DATA_DIR = _BASE / 'data' / 'processed'
LOGS_DIR           = _BASE / 'logs'


#  数据用
WINDOW_SIZE = 168
INPUT_SIZE = 15
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

#  模型用
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2

QUANTILES = [0.1, 0.5, 0.9]
NUM_QUANTILES = len(QUANTILES)

#  训练用
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 150
PATIENCE = 20