import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.batch_test import *
from utils.helper import *
import os
import sys
import torch.optim as optim
from time import time


class SetRank(nn.Module):

    def __init__(self, userNum, itemNum):
        super(SetRank, self).__init__()
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

    def forward(self, users, pos_items, pot_items, neg_items, drop_flag=False):
        user_embedding = self.embedding_dict['user_emb'][users, :]
        pos_i_embedding = self.embedding_dict['item_emb'][pos_items, :]
        pot_i_embedding = self.embedding_dict['item_emb'][pot_items, :]
        neg_i_embedding = self.embedding_dict['item_emb'][neg_items, :]
        return user_embedding, pos_i_embedding, pot_i_embedding, neg_i_embedding

    def rating(self, user_embedding, pos_i_embedding):
        return torch.matmul(user_embedding, pos_i_embedding.t())

    def create_bpr_loss(self, users, pos_items, pot_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        pot_scores = torch.sum(torch.mul(users, pot_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        maxi_1 = nn.LogSigmoid()(pos_scores - neg_scores)
        maxi_2 = nn.LogSigmoid()(pos_scores - pot_scores)
        maxi_3 = nn.LogSigmoid()(pot_scores - neg_scores)

        mf_loss = -1 * (torch.mean(maxi_1) + torch.mean(maxi_2) + torch.mean(maxi_3))

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 +
                       torch.norm(pot_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss


    def create_list_loss(self, users, pos_items, neg_items):
        users_embeddings = self.embedding_dict['user_emb'][users, :]
        pos_embeddings = self.embedding_dict['item_emb'][pos_items, :]
        neg_embeddings = self.embedding_dict['item_emb'][neg_items, :]

        score_ui = torch.exp(torch.sum(torch.mul(users_embeddings, pos_embeddings), axis = 1))
        users_embeddings_new = torch.unsqueeze(users_embeddings, dim=1).expand(neg_embeddings.size())
        score_uj = torch.sum(torch.exp(torch.sum(torch.mul(users_embeddings_new, neg_embeddings), dim=2)), dim=1)
        list_loss = torch.mean(torch.log(score_ui / (score_ui + score_uj)) * -1)
        regularizer = (torch.norm(users_embeddings) ** 2 + torch.norm(pos_embeddings) ** 2 + \
                       torch.norm(neg_embeddings) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return list_loss + emb_loss, list_loss, emb_loss






if __name__ == "__main__":
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print(torch.cuda.is_available())
    # print(args.device)

    model = SetRank(data_generator.n_users,
                  data_generator.n_items).to(args.device)
    t0 = time()
    #Train
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    for epoch in range(args.epoch):
        t1 = time()
        loss, list_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):

            users, pos_items, neg_items_QR = data_generator.sample_LPR()

            batch_loss, batch_list_loss, batch_emb_loss = model.create_list_loss(users, pos_items,
                                                                                     neg_items_QR)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            list_loss += batch_list_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, list_loss, emb_loss)
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
                       (epoch, t2 - t1, t3 - t2, loss, list_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
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
