import os
from torch.utils.tensorboard import SummaryWriter
import logging


class Recorder:
    def __init__(self,
                 graphic_save_path,
                 performance_save_path,
                 logging_save_path):
        self.graphic_save_path = graphic_save_path
        self.performance_save_path = performance_save_path
        self.logging_save_path = logging_save_path
        self.early_stopping_epoch = None
        self.train_loss = []
        self.valid_performance = []
        self.writer = SummaryWriter(performance_save_path)
        self.logger = None

    def check_dir_exist(self):
        """
        Check whether all saving path exist, create them if not.
        """
        if not os.path.exists(self.graphic_save_path):
            os.mkdir(self.graphic_save_path)
        if not os.path.exists(self.performance_save_path):
            os.mkdir(self.performance_save_path)

    def create_logger(self):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        # create logging instance
        self.logger = logging.getLogger()

        # logging to file
        logger_handler = logging.FileHandler(filename=self.logging_save_path,
                                             mode='a',
                                             encoding='utf-8')
        self.logger.addHandler(logger_handler)

    def add_train_loss(self, train_loss):
        self.train_loss.append(train_loss)

    def add_valid_performance(self, valid_loss):
        self.valid_performance.append(valid_loss)

    def record_early_stopping_epoch(self, epoch):
        self.early_stopping_epoch = epoch

    def get_writer(self):
        return self.writer

    def get_train_loss(self):
        return self.train_loss

    def get_valid_performance(self):
        return self.valid_performance

    def get_early_stopping_epoch(self):
        return self.early_stopping_epoch

    def get_logger(self):
        return self.logger
