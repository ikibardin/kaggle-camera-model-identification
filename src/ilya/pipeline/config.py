import multiprocessing

MODELS = [
    'densenet201_d4',
    'dpn92',
    'densenet201',
    'resnext101',
    'densenet161',
    'se_resnext50',
    'se_resnet50',
    'dpn98',
    'resnext101_d4'
]

CROP_SIZE = 480
STEPS = 1000
INITIAL_LR = 1e-4
PATIENCE = 3
NUM_WORKERS = multiprocessing.cpu_count()

LOGS_DIR = '../../logs'

CONFIG_LOG_MESSAGE = 'crop_size={}; steps={}; initial_lr={}; patience={}; \'' \
                     'n_workers={}; use_d4={}'

INIT_CYCLIC_LR = 1e-3
CYCLE_LEN = 10

BATCH_SIZES = {
    'densenet201_d4': 40,
    'dpn92': 48,
    'densenet201': 40,
    'resnext101': 28,
    'densenet161': 32,
    'se_resnext50': 56,
    'se_resnet50': 64,
    'dpn98': 32,
    'resnext101_d4': 28
}

TABLES_DIR = '../../tables'

CHECKPOINTS_DIR = '../../checkpoints/{}'

DATASET_DIR = '../../data'

TEST_DIR = DATASET_DIR + '/test'

PSEUDO_DIR = DATASET_DIR + '/pseudo'
