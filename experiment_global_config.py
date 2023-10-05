from pathlib import Path

# -------------------------------------
FOLDER_PATH = Path("experiments")
AUTHOR = "Author"

# -------------------------------------
N_KERNELS = -1
PARALLEL = True
# -------------------------------------
EXPERIMENT = "Synthetic data"
K = 5

# EXPERIMENT = "MNIST"
# K = 20

# EXPERIMENT = "FEMNIST"
# K = 64

# -------------------------------------
N_CLIENTS = 100  # Ignored for FEMNIST
N_CLIENTS_PER_ROUND = range(5, 101, 5)
TOL_GLOBAL = 10 ** (-8)
TOL_LOCAL = 10 ** (-8)
LEARNING_RATE = 0.01
MOMENTUM = 0.8
STEPS_WITHOUT_IMPROVEMENTS = 300
REPETITIONS = 100
MAX_ITER = 10_000
