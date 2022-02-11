from collections import defaultdict
from time import time

from datasketch import MinHash, MinHashLSHForest
from tqdm import tqdm

def get_interacted(data_file):
    data_file = data_file
    n_users, n_items = 0, 0
    interacted = {}
    with open(data_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l]
                uid, interacted_items = items[0], items[1:]
                interacted[uid] = interacted_items

                n_items = max(n_items, max(items))
                n_users = max(n_users, uid)
    n_items += 1
    n_users += 1
    return interacted, n_users, n_items


def get_minhash(items):
    m = MinHash(num_perm=128)
    for item in items:
        m.update(str(item).encode('utf8'))
    return m


def compute_jaccard(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    return len(set1 & set2) / len(set1 | set2)

def get_sim(interacted, k):
    t1 = time()
    # 先构建minhash集合
    u_forest = MinHashLSHForest(num_perm=128)
    for u, items in tqdm(interacted.items()):
        m_hash = str(u)
        m = MinHash(num_perm=128)
        for item in items:
            m.update(str(item).encode('utf8'))
        u_forest.add(m_hash, m)
    u_forest.index()

    # 计算相似度
    W = {}

    for u, items in tqdm(interacted.items()):
        m_u = get_minhash(items)
        result = u_forest.query(m_u, 2 * k)
        result = [(key, compute_jaccard(interacted[u], interacted[int(key)])) for key in result]
        result = sorted(result, key=lambda x: x[1], reverse=True)[1:k]
        W[u] = {}
        for item in result:
            v = int(item[0])
            sim = item[1]
            W[u][v] = sim
    print("Time of cal sim:{}".format(time() - t1))
    return W


def recommend(user, interacted, W, data_file):
    rank = defaultdict(int)
    interacted_items = interacted[user]
    freq, n_users = get_freq(data_file)
    for v, wuv in W[user].items():
        for i in interacted[v]:
            if i in interacted_items:
                continue
            rank[i] += wuv
    return rank


def get_potential_list(path, alpha, beta):
    (alpha, beta) = alpha, beta
    data_file = path + "data.txt"
    pot_file = path + "potential"+".txt"
    interacted, n_users, n_items = get_interacted(data_file)
    k = int(n_users * alpha)
    p_k = int(n_items * beta)
    W = get_sim(interacted, k)
    potential = open(pot_file, "a")
    for u, items in tqdm(interacted.items()):
        potential.write(str(u))
        rank = recommend(u, interacted, W, data_file)
        p_list = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:p_k]
        for p_item in p_list:
            potential.write(" " + str(p_item[0]))
        potential.write("\n")
    potential.close()

def get_freq(data_file):
    freq = {}
    n_users = 0
    with open(data_file) as f_d:
        for l in f_d.readlines():
            l = l.strip('\n').split(' ')
            uid = int(l[0])
            n_users = max(n_users, uid)
            items = [int(i) for i in l[1:]]
            for item in items:
                cnt = freq.get(item, 0)
                freq[item] = cnt + 1
    return freq, n_users + 1


if __name__ == "__main__":

    pre = "../../data/"
    path = pre + "lastfm-2k/"
    alpha, beta = 0.025, 0.025
    get_potential_list(path, alpha, beta)


