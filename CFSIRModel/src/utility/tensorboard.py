import os
from torch.utils.tensorboard import SummaryWriter

class TensorBoard:

    @staticmethod
    def get_tensorboard_writer(log_dir="runs"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return SummaryWriter(log_dir) 