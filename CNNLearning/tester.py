import os

import numpy as np

from tools.trainDataset import MyDataset
from torch.utils.data import DataLoader
from dataHandler.CIFAR10Handler import CIFAR10Handler
import torch
from tqdm import tqdm
from tools.recorder import Recorder
from nets import MODEL


class Tester:
    def __init__(self,
                 label_name_file,
                 data_file,
                 test_data_file,
                 in_channel,
                 height,
                 width,
                 class_num,
                 model_name,
                 batch_size,
                 lr,
                 dropout,
                 gpu,
                 seed):
        # file path
        self.label_name_file = label_name_file
        self.data_file = data_file
        self.test_data_file = test_data_file
        self.test_performance_path = None
        self.test_logger_path = None
        self.test_graphic_path = None
        self.test_model_path = None

        # experiment settings
        self.in_channel = in_channel
        self.height = height
        self.width = width
        self.class_num = class_num
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout

        # testing settings
        self.gpu = gpu
        self.seed = seed
        self.recorder = None
        self.logger = None
        self.writer = None
        self.network = None

    def set_save_path(self):
        path = os.getcwd()
        path = os.path.join(path, 'results')
        path = os.path.join(path, self.model_name)
        path = os.path.join(path, 'batchSize_' + str(self.batch_size) +
                            '_lr_' + str(self.lr) +
                            '_dropout_' + str(self.dropout))
        path = os.path.join(path, str(self.seed))
        path = os.path.join(path, 'test')
        self.test_performance_path = os.path.join(path, 'performance')
        self.test_graphic_path = os.path.join(path, 'graphic')
        self.test_logger_path = os.path.join(path, 'test.log')
        path = os.path.join(path, '..')
        path = os.path.join(path, 'train')
        self.test_model_path = os.path.join(path, 'parameter.pkl')

    def create_recorder(self):
        self.recorder = Recorder(self.test_graphic_path,
                                 self.test_performance_path,
                                 self.test_logger_path)
        self.recorder.create_logger()

    def set_device(self):
        gpu = str(self.gpu)
        if torch.cuda.is_available():
            self.logger.info('==> use gpu id: {}'.format(gpu))
            self.device = torch.device('cuda:' + gpu)
        else:
            self.logger.warning('No GPU found or Wrong GPU id, using CPU instead.')
            self.device = torch.device('cpu')

    def load_model(self, model_path=None):
        model_path = self.test_model_path if model_path is None else model_path
        self.network = MODEL[self.model_name](
            self.in_channel,
            self.width,
            self.height,
            self.class_num,
            self.dropout
        ).to(self.device)
        self.logger.info('network structure: ' + str(self.network))
        self.network.load_state_dict(torch.load(model_path))

    def test(self, model_path):
        # set saving path
        self.set_save_path()

        # create recorder
        self.create_recorder()

        # create logger
        self.logger = self.recorder.get_logger()

        # get summary writer
        self.writer = self.recorder.get_writer()

        # using GPU
        self.set_device()

        # get test dataset
        self.logger.info('==> Get test dataset')
        handler = CIFAR10Handler(self.label_name_file, self.data_file, self.test_data_file)
        handler.extract_data()
        test_data, test_label = handler.get_test_dataset()
        test_dataset = MyDataset(test_data, test_label)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        self.logger.info(str(handler))

        # loader network
        self.logger.info('==> Load network')
        self.load_model(model_path)
        self.logger.info('network structure: ' + str(self.network))

        # testing
        self.network.eval()
        self.logger.info('==> Testing start')
        true_count = 0
        for data in tqdm(test_loader):
            with torch.no_grad():
                output = self.network(data[0].float().to(self.device)).cpu().detach().numpy()
                true_count += np.sum(np.argmax(output, axis=1) == data[1].numpy())
        val_precision = true_count / len(test_label)
        self.writer.add_scalar('test/percision', val_precision)
        self.logger.info('Precision: ' + str(val_precision * 100) + '%')

        self.logger.info('==> Testing complete')
