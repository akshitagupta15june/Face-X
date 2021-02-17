
def init_random(seed=0):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)

