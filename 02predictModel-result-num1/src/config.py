'''
超参数配置
'''
from pathlib import Path


RAW_DATA_DIR=Path(__file__).parent.parent / "data"/"raw"
MODELS_DIR=Path(__file__).parent.parent / "models"
PROCESSED_DATA_DIR=Path(__file__).parent.parent / "data"/"processed"
LOGS_DIR=Path(__file__).parent.parent / "logs"


#  数据用
WINDOW_SIZE = 168
INPUT_SIZE = 15
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

#  模型用
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2

#  训练用
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 150
PATIENCE = 20