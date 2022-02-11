from collections import defaultdict

import numpy as np
import pandas as pd

path = "../../data/lastfm-2k/"

def get_map():
    movies = pd.read_csv(path + "movies.dat", sep="::", header=None, names=["MovieID", "Title", "Genres"], engine='python')
    users = pd.read_csv(path + "users.dat", sep="::", header=None, names=["UserID", "Gender", "Age", "Occupation", "Zipcode"], engine='python')
    map_dict = {}

    for index in range(len(movies)):
        if index + 1 != movies.iloc[index].MovieID:
            map_dict[movies.iloc[index].MovieID] = index + 1

    def fun1(key):
        if key in map_dict.keys():
            return map_dict[key]
        else:
            return key

    movies['newID'] = movies['MovieID'].apply(lambda x: fun1(x))
    movies.drop(['MovieID'], inplace=True, axis=1)
    movies = movies.rename(columns={'newID': 'MovieID'})
    movies = movies[['MovieID', 'Title', 'Genres']]
    ratings = pd.read_csv(path + "ratings.dat", sep="::", header=None, names=["UserID", "ItemID", "rating", "timestamp"], engine='python')
    ratings['NewID'] = ratings['ItemID'].apply(lambda x: fun1(x))
    ratings = ratings[['UserID', 'NewID']]
    ratings = ratings.rename(columns={'NewID': 'MovieID'})
    users.to_csv(path + 'users.csv', index=False)
    movies.to_csv(path + 'movies.csv', index=False)
    ratings.to_csv(path + 'ratings.csv', index=False)

def get_data():
    ratings = pd.read_csv(path + "ratings.csv")
    interacted = defaultdict(list)
    for i in range(len(ratings)):
        tmp = ratings.iloc[i]
        uid, mid = tmp['UserID'] - 1, tmp['MovieID'] - 1
        interacted[uid].append(mid)
    data = open(path + "data.txt", "a")
    for key in interacted:
        movies = interacted[key]
        data.write(str(key))
        for movie in movies:
            data.write(" " + str(movie))
        data.write("\n")
    data.close()


def split_data():
    train = open(path + "train.txt", "a")
    test = open(path + "test.txt", "a")
    train_rate = 0.8

    with open(path + "data.txt") as f:
        for l in f.readlines():
            l = l.strip("\n").split(" ")
            train.write(l[0])
            test.write(l[0])
            items = l[1:]
            for index, item in enumerate(items):
                #先往测试集写一个，避免测试集为0的情况
                if index == 0:
                    test.write(" " + item)
                    continue
                rate = np.random.rand(1)
                if(rate < train_rate):
                    train.write(" " + item)
                else:
                    test.write(" " + item)
            train.write("\n")
            test.write("\n")

    train.close()
    test.close()

def split_validate_data():
    datasets = ["amazon-book", "ciao", "citeulike", "epinions", "gowalla", "lastfm-2k", "ml-1m", "yelp2018"]
    for dataset in datasets:
        path = "../../data/" + dataset
        train = open(path + "/train_val.txt", "a")
        test = open(path + "/val.txt", "a")
        train_rate = 0.9

        with open(path + "/train.txt") as f:
            for l in f.readlines():
                l = l.strip("\n").split(" ")
                train.write(l[0])
                test.write(l[0])
                items = l[1:]
                for index, item in enumerate(items):
                    #先往测试集写一个，避免测试集为0的情况
                    if index == 0:
                        test.write(" " + item)
                        continue
                    rate = np.random.rand(1)
                    if(rate < train_rate):
                        train.write(" " + item)
                    else:
                        test.write(" " + item)
                train.write("\n")
                test.write("\n")

        train.close()
        test.close()

if __name__ == "__main__":
    # get_map()
    # get_data()
    # split_data()
    split_validate_data()