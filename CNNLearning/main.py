import random
from trainer import Trainer
from tester import Tester
import argparse


parser = argparse.ArgumentParser(description='CNN experiment')
parser.add_argument('--model', type=str, default='LeNet5', help='[SL_CNN, LeNet5]')
parser.add_argument('--inChannel', type=int, default=3, help='input channel')
parser.add_argument('--height', type=int, default=32, help='height of image')
parser.add_argument('--width', type=int, default=32, help='width of image')
parser.add_argument('--classNum', type=int, default=10, help='number of classes')
parser.add_argument('--batchSize', type=int, default=48, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--optimizerWeightDecay', type=float, default=1e-05, help='L2 penalty')
parser.add_argument('--earlyStoppingPatience', type=int, default=5, help='tolerance of early stopping')
parser.add_argument('--earlyStoppingValInterval', type=int, default=10, help='interval between two validating')
parser.add_argument('--earlyStoppingDelta', type=float, default=0.001, help='criterion of violation')
parser.add_argument('--maxEpoch', type=int, default=10000, help='maximum epoch')
parser.add_argument('--nRound', type=int, default=1, help='training round')
parser.add_argument('--gpu', type=int, default=0, help='gpu ids (default: 0)')
opt = parser.parse_args()


if __name__ == '__main__':
    params = dict()
    params['in_channel'] = opt.inChannel
    params['height'] = opt.height
    params['width'] = opt.width
    params['class_num'] = opt.classNum
    params['model_name'] = opt.model
    params['batch_size'] = opt.batchSize
    params['lr'] = opt.lr
    params['dropout'] = opt.dropout
    params['optimizer_weight_decay'] = opt.optimizerWeightDecay
    params['early_stopping_patience'] = opt.earlyStoppingPatience
    params['early_stopping_interval'] = opt.earlyStoppingValInterval
    params['early_stopping_delta'] = opt.earlyStoppingDelta
    params['maxEpoch'] = opt.maxEpoch
    params['gpu'] = opt.gpu
    params['label_name_file'] = r'data/cifar-10-batches-py/batches.meta'
    params['data_file'] = [
        r'data/cifar-10-batches-py/data_batch_1',
        r'data/cifar-10-batches-py/data_batch_2',
        r'data/cifar-10-batches-py/data_batch_3',
        r'data/cifar-10-batches-py/data_batch_4',
        r'data/cifar-10-batches-py/data_batch_5'
    ]
    params['test_data_file'] = r'data/cifar-10-batches-py/test_batch'

    for r in range(opt.nRound):
        print('-------------------------------------------------')
        print('-------------------- ' + str(r + 1) + ' round --------------------')
        print('-------------------------------------------------')
        params['seed'] = random.randint(0, 1000)
        trainer = Trainer(**params)
        trainer.train()
        tester = Tester(**params)
        tester.test()
