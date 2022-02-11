import numpy as np
import scipy.sparse as sp
import random as rd
from time import time
from utils.parser import parse_args
import utils.generate_pot as generate_pot
import os
args = parse_args()


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        potential_file = path + '/potential.txt'
        data_file = path + '/data.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        #self.k is the number of sampling
        self.k = args.k

        #Frequency of each item
        self.freq = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    if len(items) == 0:
                        continue
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    if len(items) == 0:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        self.potential_items = {}
        if not os.path.exists(potential_file):
            print("Generate potential preference items: ")
            generate_pot.get_potential_list(path + '/', 0.025, 0.025)
        with open(potential_file) as f_p:
            for l in f_p.readlines():
                l = l.strip('\n').replace('\x00', '')
                items = [int(i) for i in l.split(' ')]
                uid, potential_items = items[0], items[1:]
                self.potential_items[uid] = potential_items


    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def sample_MulSetRank(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                # pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_id = rd.randint(0, n_pos_items-1)
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_pot_items_for_u(u, num):
            # sample num pot items for u-th user
            pot_items = self.potential_items[u]
            n_pot_items = len(pot_items)
            if n_pot_items == 0:
                return [0]
            pot_batch = []
            while True:
                if len(pot_batch) == num:
                    break
                # pot_id = np.random.randint(low=0, high=n_pot_items, size=1)[0]
                pot_id = rd.randint(0, n_pot_items-1)
                pot_i_id = pot_items[pot_id]

                if pot_i_id not in pot_batch:
                    pot_batch.append(pot_i_id)
            return pot_batch

        def sample_neg_items_for_u_R(u, num):
            # sample num neg items for u-th user
            # sampling based
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = rd.randint(0, self.n_items - 1)
                if neg_id not in self.train_items[u] and neg_id not in neg_items and neg_id not in self.potential_items[
                    u]:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_Q(u, num):
            # sampling based
            if num > len(self.potential_items[u]):
                return self.potential_items[u] + [i for i in range(num - len(self.potential_items[u]))]
            neg_items = rd.sample(self.potential_items[u], num)
            return neg_items

        pos_items, pot_items, neg_items_Q, neg_items_R = [], [], [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            pot_items += sample_pot_items_for_u(u, 1)
            neg_items_Q.append(sample_neg_items_for_u_Q(u, int(self.n_items/2 * 0.025)))  # PQ
            neg_items_R.append(sample_neg_items_for_u_R(u, self.k))  # R
        return users, pos_items, pot_items, neg_items_Q, neg_items_R

    def sample_BPRMF(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                # pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_id = rd.randint(0, n_pos_items-1)
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                # neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                neg_id = rd.randint(0, self.n_items-1)
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u_DNS(u, 1, 5, model)
        return users, pos_items, neg_items

    def sample_CPR(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                # pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_id = rd.randint(0, n_pos_items-1)
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_pot_items_for_u(u, num):
            # sample num pot items for u-th user
            pot_items = self.potential_items[u]
            n_pot_items = len(pot_items)
            if n_pot_items == 0:
                return [0]
            pot_batch = []
            while True:
                if len(pot_batch) == num:
                    break
                # pot_id = np.random.randint(low=0, high=n_pot_items, size=1)[0]
                pot_id = rd.randint(0, n_pot_items-1)
                pot_i_id = pot_items[pot_id]

                if pot_i_id not in pot_batch:
                    pot_batch.append(pot_i_id)
            return pot_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                # neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                neg_id = rd.randint(0, self.n_items-1)
                if neg_id not in self.train_items[u] and neg_id not in neg_items and neg_id not in self.potential_items[
                    u]:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, pot_items, neg_items = [], [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            pot_items += sample_pot_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u_DNS(u, 1, 5, model)
        return users, pos_items, pot_items, neg_items

    def sample_BPRDNS(self, model):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                # pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_id = rd.randint(0, n_pos_items-1)
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u_DNS(u, num, k, model):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                jsam = 0
                jscore = 0
                cnt = 0
                neg_ls = []
                while cnt < k:
                    # neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                    neg_id = rd.randint(0, self.n_items-1)
                    if neg_id not in self.train_items[u] and neg_id not in neg_items:
                        cnt += 1
                        neg_ls.append(neg_id)
                # t_1 = time()
                u_embedding = model.embedding_dict['user_emb'][u].cpu().detach().numpy()
                i_embeddings = model.embedding_dict['item_emb'][neg_ls].cpu().detach().numpy()
                predict_scores = np.dot(i_embeddings, u_embedding)
                # print(time() - t_1)
                max_index = np.argmax(predict_scores)
                jsam = neg_ls[max_index]
                neg_items.append(jsam)

            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u_DNS(u, 1, 2, model)
        return users, pos_items, neg_items

    def sample_LPR(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                # pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_id = rd.randint(0, n_pos_items-1)
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u_QR(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                # neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                neg_id = rd.randint(0, self.n_items-1)
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items_QR = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items_QR.append(sample_neg_items_for_u_QR(u, 5))
        return users, pos_items, neg_items_QR
