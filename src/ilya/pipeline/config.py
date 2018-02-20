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
    'densenet201_d4': 20,
    'dpn92': 24,
    'densenet201': 20,
    'resnext101': 14,
    'densenet161': 16,
    'se_resnext50': 28,
    'se_resnet50': 36,
    'dpn98': 16,
    'resnext101_d4': 14
}

TABLES_DIR = '../../tables'

CHECKPOINTS_DIR = '../../checkpoints/{}'

DATASET_DIR = '../../dataset'

TEST_DIR = DATASET_DIR + '/test'

PSEUDO_DIR = DATASET_DIR + '/pseudo'
