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
NUM_WORKERS = 8

LOGS_DIR = '../../logs'

CONFIG_LOG_MESSAGE = 'crop_size={}; steps={}; initial_lr={}; patience={}; \'' \
                     'n_workers={}; use_d4={}'

INIT_CYCLIC_LR = 1e-3
CYCLE_LEN = 10

BATCH_SIZES = {
    'dpn92': 12,
}

CHECKPOINTS_DIR = '../../checkpoints/{}'
