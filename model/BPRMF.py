import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.batch_test import *
from utils.helper import *
import os
import sys
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
from time import time


class BPRMF(nn.Module):

    def __init__(self, userNum, itemNum):
        super(BPRMF, self).__init__()
        self.n_users = userNum
        self.n_items = itemNum
        self.device = args.device
        self.lr = args.lr
        self.embed_size = args.embed_size
        self.batch_size = args.batch_size

        self.decay = eval(args.regs)[0]

        self.embedding_dict = self.init_weight()

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users,
                                                 self.embed_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items,
                                                 self.embed_size)))
        })
        return embedding_dict

    def forward(self, users, pos_items, neg_items, _, drop_flag=False):
        user_embedding = self.embedding_dict['user_emb'][users, :]
        pos_i_embedding = self.embedding_dict['item_emb'][pos_items, :]
        neg_i_embedding = self.embedding_dict['item_emb'][neg_items, :]
        return user_embedding, pos_i_embedding, neg_i_embedding, _

    def rating(self, user_embedding, pos_i_embedding):
        return torch.matmul(user_embedding, pos_i_embedding.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)


        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss


if __name__ == "__main__":
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BPRMF(data_generator.n_users,
                  data_generator.n_items).to(args.device)
    t0 = time()
    #Train
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample_BPRMF()
            users = torch.LongTensor(users).cuda()
            pos_items = torch.LongTensor(pos_items).cuda()
            neg_items = torch.LongTensor(neg_items).cuda()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, _ = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           [])

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break


    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)