__version__ = "0.1.0"
import logzero

formatter = logzero.LogFormatter(fmt="%(color)s[%(levelname)1.1s %(asctime)s]%(end_color)s %(message)s")
logzero.setup_default_logger(formatter=formatter)
import random
import numpy as np
import torch
import tensorflow as tf

torch.backends.cudnn.enabled = False

seed = 10
# random
random.seed(seed)
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# Tensorflow
tf.random.set_seed(seed)
