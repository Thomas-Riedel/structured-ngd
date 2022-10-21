from torch.optim import Optimizer


class NoisyOptimizer(Optimizer):
    def __init__(self, params, defaults, *args, **kwargs):
        super(NoisyOptimizer, self).__init__(params, defaults)
