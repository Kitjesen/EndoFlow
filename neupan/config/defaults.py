"""
Default hyperparameters and configuration constants.
"""

# MPC defaults
RECEDING = 10
STEP_TIME = 0.1
REF_SPEED = 4.0
DEVICE = "cpu"

# PAN defaults
PAN_ITER_NUM = 2
PAN_ITER_THRESHOLD = 0.1
DUNE_MAX_NUM = 100
NRMP_MAX_NUM = 10

# Adjust / cost-weight defaults
Q_S = 1.0          # state cost weight
P_U = 1.0          # speed cost weight
ETA = 10.0          # slack gain (L1)
D_MAX = 1.0         # max safety distance
D_MIN = 0.1         # min safety distance
RO_OBS = 400        # collision avoidance penalty
BK = 0.1            # proximal coefficient
SOLVER = "ECOS"

# DUNE training defaults
TRAIN_DATA_SIZE = 100000
TRAIN_DATA_RANGE = [-25, -25, 25, 25]
TRAIN_BATCH_SIZE = 256
TRAIN_EPOCH = 5000
TRAIN_LR = 5e-5
TRAIN_LR_DECAY = 0.5
TRAIN_DECAY_FREQ = 1500
TRAIN_VALID_FREQ = 100
TRAIN_SAVE_FREQ = 500

# Collision
COLLISION_THRESHOLD = 0.1
