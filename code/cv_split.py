import numpy as np

class UnseenTestSplit:
    def __init__(self, n_splits = 5, class_per_split = 3):
        self.n_splits = n_splits
        self.class_per_split = class_per_split

    def split(self, X, y, groups=None):
        classes = np.unique(y)
        classes = np.array([i for i in classes if i != 0])

        all_index = np.arange(len(y))

        for i in range(self.n_splits):
            train_index = 0
            test_index = 0
            np.random.shuffle(classes)
            test_classes = classes[:self.class_per_split]

            print(test_classes)
            test_index = all_index[np.isin(y,test_classes)]
            train_index = all_index[~np.isin(y,test_classes)]

            train_index = np.append(train_index, all_index[y == 0])
            np.random.shuffle(train_index)
            test_index = np.append(test_index, train_index[int(len(train_index)*0.8):])
            train_index = train_index[:int(len(train_index)*0.8)]
            yield train_index, test_index

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
