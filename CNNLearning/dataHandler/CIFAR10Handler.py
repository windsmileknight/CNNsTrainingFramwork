import pickle

import numpy as np


class CIFAR10Handler:
    def __init__(self,
                 label_name_file,
                 data_file,
                 test_data_file,
                 normalized=True
                 ):
        """
        Only suitable to CIFAR10 dataset, so data path are packaged in class.
        """
        self.label_name_file = label_name_file
        self.data_file = data_file
        self.test_data_file = test_data_file
        self.label_names = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.normalized = normalized

    def extract_data(self):
        """
        extract data from dataset
        :return: train_data, train_label, valid_data, valid_label, test_data, test_label
        """
        # extract label names
        with open(self.label_name_file, 'rb') as f:
            self.label_names = pickle.load(f, encoding='utf8')['label_names']

        # extract data batches for train and valid
        image_batches = []
        label_batches = []
        for path in self.data_file:
            with open(path, 'rb') as f:
                data_batch = pickle.load(f, encoding='bytes')
                image_batches.append(data_batch[b'data'])  # shape of each image batch: [10000, 3072]
                label_batches.append(data_batch[b'labels'])  # shape of each label batch: [10000,]

        image_batches = [image_batch.reshape((-1, 3, 32, 32)) for image_batch in image_batches]
        if self.normalized:
            image_batches = [self.normalization(image_batch) for image_batch in image_batches]

        # extract valid dataset from data batches
        valid_data = image_batches.pop()
        valid_label = label_batches.pop()
        self.valid_dataset = (valid_data, valid_label)

        # construct train dataset
        train_data = np.concatenate(image_batches, axis=0)  # shape of each image: [3 × 32 × 32]
        train_label = np.concatenate(label_batches, axis=0)
        self.train_dataset = (train_data, train_label)

        # extract test dataset
        with open(self.test_data_file, 'rb') as f:
            test_batch = pickle.load(f, encoding='bytes')
            test_data = test_batch[b'data'].reshape((-1, 3, 32, 32))
            test_data = self.normalization(test_data) if self.normalized else test_data
            test_label = test_batch[b'labels']
            self.test_dataset = (test_data, test_label)

    @staticmethod
    def normalization(batch):
        """
        normalize RGB image.
        TODO: not a good way, model perform worse after normalization.
        :param batch: shape: [-1, 3, 32, 32]
        :return: normalized image.
        """
        sum_per_RGB_pixel = np.expand_dims(np.sum(batch, axis=1), 1)  # [-1, 32, 32]
        return batch / (sum_per_RGB_pixel + 1e-06)  # using slight value to handle 0/0

    def get_train_dataset(self):
        """
        obtain train dataset.
        :return: tuple: (data, label)
        """
        if self.test_dataset is None:
            raise Exception('You should call extract_data() first.')
        return self.train_dataset

    def get_valid_dataset(self):
        """
        obtain valid dataset.
        :return: tuple: (data, label)
        """
        if self.test_dataset is None:
            raise Exception('You should call extract_data() first.')
        return self.valid_dataset

    def get_test_dataset(self):
        """
        obtain test dataset.
        :return: tuple: (data, label)
        """
        if self.test_dataset is None:
            raise Exception('You should call extract_data() first.')
        return self.test_dataset

    def get_label_names(self):
        """
        obtain label names.
        :return: dict
        """
        if self.test_dataset is None:
            raise Exception('You should call extract_data() first.')
        return self.label_names

    def __str__(self):
        return 'train dataset size (samples × weight × height × channel): ' + str(self.train_dataset[0].shape) + '\n'\
               + 'valid dataset size (samples × weight × height × channel): ' + str(self.valid_dataset[0].shape) + '\n'\
               + 'test dataset size (samples × weight × height × channel): ' + str(self.test_dataset[0].shape) + '\n'\
               + 'labels name: ' + str(self.label_names) + '\n'


if __name__ == '__main__':
    label_name_file = r'../data/cifar-10-batches-py/batches.meta'
    data_file = [
        r'../data/cifar-10-batches-py/data_batch_1',
        r'../data/cifar-10-batches-py/data_batch_2',
        r'../data/cifar-10-batches-py/data_batch_3',
        r'../data/cifar-10-batches-py/data_batch_4',
        r'../data/cifar-10-batches-py/data_batch_5'
    ]
    test_data_file = r'../data/cifar-10-batches-py/test_batch'
    handler = CIFAR10Handler(label_name_file, data_file, test_data_file)
    handler.extract_data()
    print(9 in handler.train_dataset[1])
