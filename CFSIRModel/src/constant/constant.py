import os 
from pathlib import Path
current_dir = Path.cwd().parent

class Constants:
    MODEL_PATH = 'src/model/cfsir_model.pth'
    DEVICE = 'DEVICE'
    CUDA = 'cuda'
    CPU = 'cpu'

    # data preprocessing constants
    DATA_BATCH = "data_batch_"
    TEST_BATCH = "test_batch"
    DATA = "data"
    LABELS = "labels"
    ENCODING = "bytes"
    FILE_OPENING_FORMAT = 'rb'

    # dataloader constants
    DATA_DIR_PATH = 'src/datapreprocessing/RawData/cifar-10-batches-py'
    BATCH_SIZE = 64

    # config file path
    CONFIG_FILE_PATH = 'src/config/config.ini'
    
    # model params
    MODEL_PARAMETERS = 'MODEL_PARAMETERS'
    EPOCHS = 'EPOCHS'
    LEARNING_RATE = 'LEARNING_RATE'
    DROPOUT_RATE = 'DROPOUT_RATE'
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    SAVED_MODEL_PATH = 'SAVED_MODEL_PATH'

    # data param
    DATASET = 'DATASET'
    DATA_DIR = 'DATA_DIR'
    BATCH_SIZE = 'BATCH_SIZE'

    # numerical constants
    range_255 = 255.0
    ZERO ,ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, \
    ELEVEN, TWELVE, THIRTEEN, FOURTEEN, FIFTEEN, SIXTEEN, SEVENTEEN, EIGHTEEN, NINETEEN, TWENTY, \
    TWENTY_ONE, TWENTY_TWO, TWENTY_THREE, TWENTY_FOUR, TWENTY_FIVE, TWENTY_SIX, TWENTY_SEVEN, TWENTY_EIGHT, TWENTY_NINE, THIRTY = range(31)
