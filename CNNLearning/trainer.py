from dataHandler.CIFAR10Handler import CIFAR10Handler
import torch
import os
from tools.trainDataset import MyDataset
from tools.recorder import Recorder
from torch.utils.data import DataLoader
from tools.earlyStopping import EarlyStopping
from nets import MODEL
import numpy as np
from tqdm import tqdm


class Trainer:
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
                 optimizer_weight_decay,
                 early_stopping_patience,
                 early_stopping_interval,
                 early_stopping_delta,
                 maxEpoch,
                 gpu,
                 seed):
        # results saving path
        self.model_save_path = None
        self.graphic_save_path = None
        self.performance_save_path = None
        self.logging_save_path = None
        self.label_name_file = label_name_file
        self.data_file = data_file
        self.test_data_file = test_data_file

        # configuration information before training
        self.model_name = model_name
        self.gpu = gpu
        self.device = None
        self.seed = seed
        self.logger = None
        self.recorder = None
        self.writer = None
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_interval = early_stopping_interval
        self.early_stopping_delta = early_stopping_delta
        self.maxEpoch = maxEpoch
        self.monitor = None

        # configuration information in training
        self.in_channel = in_channel
        self.width = width
        self.height = height
        self.class_num = class_num
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.dropout = dropout
        self.network = None

    def set_save_path(self):
        path = os.getcwd()
        path = os.path.join(path, 'results')
        path = os.path.join(path, self.model_name)
        path = os.path.join(path, 'batchSize_' + str(self.batch_size) +
                            '_lr_' + str(self.lr) +
                            '_dropout_' + str(self.dropout))
        path = os.path.join(path, str(self.seed))
        path = os.path.join(path, 'train')
        self.model_save_path = os.path.join(path, 'parameter.pkl')
        self.graphic_save_path = os.path.join(path, 'graphic')
        self.performance_save_path = os.path.join(path, 'performance')
        self.logging_save_path = os.path.join(path, 'train.log')

    def create_recorder(self):
        self.recorder = Recorder(self.graphic_save_path,
                                 self.performance_save_path,
                                 self.logging_save_path)
        self.recorder.create_logger()

    def set_device(self):
        """
        If GPU is available, train network with specific GPU. Otherwise, use CPU.
        """
        gpu = str(self.gpu)
        if torch.cuda.is_available():
            self.logger.info('==> use gpu id: {}'.format(gpu))
            self.device = torch.device('cuda:' + gpu)
        else:
            self.logger.warning('No GPU found or Wrong GPU id, using CPU instead.')
            self.device = torch.device('cpu')

    def set_random_seed(self):
        """
        Set random seed to network.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)  # sets the seed for generating random numbers
            torch.cuda.manual_seed(self.seed)  # Sets the seed for generating random numbers for the current GPU
            torch.cuda.manual_seed_all(self.seed)  # Sets the seed for generating random numbers on all GPUs

    def set_early_stopping(self):
        self.monitor = EarlyStopping(self.model_save_path,
                                     patience=self.early_stopping_patience,
                                     val_interval=self.early_stopping_interval,
                                     delta=self.early_stopping_delta,
                                     verbose=True)

    def save_model(self):
        torch.save(self.network.state_dict(), self.model_save_path)

    def train(self):
        # create saving path
        self.set_save_path()

        # create recorder
        self.create_recorder()

        # create logger
        self.logger = self.recorder.get_logger()

        # get summary writer
        self.writer = self.recorder.get_writer()

        # create early stopping
        self.set_early_stopping()
        self.logger.info('Early stopping: patience = {}, val_interval = {}, delta = {}'.format(self.monitor.patience,
                                                                                               self.monitor.val_interval,
                                                                                               self.monitor.delta))

        # using GPU
        self.set_device()

        # set random seed
        self.set_random_seed()

        # get supervised dataset
        self.logger.info('==> Obtain supervised dataset')
        handler = CIFAR10Handler(self.label_name_file, self.data_file, self.test_data_file, normalized=False)
        handler.extract_data()
        train_data, train_label = handler.get_train_dataset()
        valid_data, valid_label = handler.get_valid_dataset()
        train_dataset = MyDataset(train_data, train_label)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataset = MyDataset(valid_data, valid_label)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        self.logger.info(str(handler))

        # build network
        self.logger.info('==> Build network')
        self.network = MODEL[self.model_name](
            self.in_channel,
            self.width,
            self.height,
            self.class_num,
            self.dropout
        ).to(self.device)
        self.logger.info('network structure: ' + str(self.network))

        # build loss, optimizer
        self.logger.info('==> Build loss and optimizer')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr,
            weight_decay=self.optimizer_weight_decay
        )

        # training
        self.logger.info('==> Training with validation dataset')
        best_epoch = 0
        for epoch in range(1, self.maxEpoch):
            epoch_loss = 0
            self.network.train()
            for data in train_loader:
                optimizer.zero_grad()
                output = self.network(data[0].float().to(self.device))
                loss = criterion(output.float().to(self.device), data[1].long().to(self.device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                torch.cuda.empty_cache()
            self.writer.add_scalar('train/train loss', epoch_loss / len(train_loader), epoch)
            self.recorder.add_train_loss(epoch_loss / len(train_loader))
            self.logger.info('Epoch: {}; loss: {}'.format(str(epoch), str(epoch_loss / len(train_loader))))

            # validating
            self.network.eval()
            true_count = 0
            for data in valid_loader:
                with torch.no_grad():
                    valid_output = self.network(data[0].float().to(self.device)).cpu().detach().numpy()
                    true_count += np.sum(np.argmax(valid_output, axis=1) == data[1].numpy())
            val_precision = true_count / len(valid_label)
            self.writer.add_scalar('train/valid precision', val_precision, epoch)
            self.recorder.add_valid_performance(val_precision)
            self.logger.info('Epoch: {}; val_precision: {}'.format(str(epoch), str(val_precision)))

            # save model periodically
            if epoch % self.early_stopping_interval == 0:
                self.save_model()

            # early stopping
            best_epoch = self.monitor(val_precision, self.network, epoch)
            if self.monitor.early_stop is True:
                self.logger.info('==> Early stopping!')
                break

        self.logger.info('==> Best epoch has been learned, which is {}'.format(str(best_epoch)))

        ################################################################################################################

        # combine train and valid dataset
        self.logger.info('==> Combine train and valid dataset')
        train_data = np.concatenate([train_data, valid_data], axis=0)
        train_label = np.concatenate([train_label, valid_label], axis=0)
        train_dataset = MyDataset(train_data, train_label)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # build new network
        self.logger.info('==> Build new network')
        self.network = MODEL[self.model_name](
            self.in_channel,
            self.width,
            self.height,
            self.class_num,
            self.dropout
        ).to(self.device)

        # build new loss, optimizer
        self.logger.info('==> Build new loss and optimizer')
        new_criterion = torch.nn.CrossEntropyLoss()
        new_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr,
            weight_decay=self.optimizer_weight_decay
        )

        # training basing on best epoch
        self.logger.info('==> Training with combined training dataset')
        self.network.train()
        for epoch in tqdm(range(1, best_epoch + 1)):
            epoch_loss = 0
            for data in train_loader:
                new_optimizer.zero_grad()
                output = self.network(data[0].float().to(self.device))
                loss = new_criterion(output.float().to(self.device), data[1].long().to(self.device))
                loss.backward()
                new_optimizer.step()
                epoch_loss += loss.item()
                torch.cuda.empty_cache()
            self.writer.add_scalar('train/train loss in best epoch', epoch_loss / len(train_loader), epoch)
            if epoch % 10 == 0:
                self.logger.info('Epoch: {}; loss: {}'.format(str(epoch), str(epoch_loss / len(train_loader))))
                self.save_model()

        self.save_model()
        self.logger.info('==> Training complete!')
