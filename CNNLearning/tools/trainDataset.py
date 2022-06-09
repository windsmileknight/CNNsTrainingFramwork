from torch.utils.data import Dataset

'''
自定义 dataset 实现 batch
'''


class MyDataset(Dataset):
    def __init__(self, train_x, train_y):
        super(MyDataset, self).__init__()

        self.train_x, self.train_y = train_x, train_y

    def __getitem__(self, item):
        return self.train_x[item], self.train_y[item]

    def __len__(self):
        return len(self.train_x)
