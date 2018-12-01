import numpy as np
import random as rd


class Data(object):
    def __init__(self, train_file, test_file):
        #get number of users and items
        self.n_users, self.n_items = 0, 0


        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    self.n_users += 1
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]
                    self.n_items = max(self.n_items, max(items))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]
                    self.n_items = max(self.n_items, max(items))
        self.n_items += 1
        self.R = np.zeros((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid][i] = 1

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

