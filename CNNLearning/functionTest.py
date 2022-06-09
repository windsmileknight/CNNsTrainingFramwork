import pickle


def test_loader():
    with open(r'data/cifar-10-batches-py/test_batch', 'rb') as f:
        label_name = pickle.load(f, encoding='bytes')
        print(label_name)
