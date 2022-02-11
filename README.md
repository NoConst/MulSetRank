# MulSetRank
This is our PyTorch implementation for the paper:

> MulSetRank: Multiple Set Ranking for Personalized Recommendation from Implicit Feedback

## Introduction

Multiple Set Ranking (MulSetRank) is a novel framework for personalized recommendation. This repository include our implementation of MulSetRank. We also implementation all baseline methods we used to compare with except NGCF. The code of NGCF can be found [here](https://github.com/xiangwang1223/neural_graph_collaborative_filtering).

## Dataset

We provide all processed datasets used in the paper.

- data.txt
  - Each line is a user with his/her positive interactions with items: userID and a list of itemID.
  - This file was user to generate potential preference items for every user and split to training and testing set.
- train.txt
  - Train file.
  - Each line is a user with his/her positive interactions with items: userID and a list of itemID.
- test.txt
  - Test file
  - Each line is a user with his/her positive interactions with items: userID and a list of itemID.

## Environment Requirement

- Python 3.5.2
- PyTorch 1.5.1

## Example to run the codes.

Run MulSetRank:

```python
cd model/
python3 MulSetRank.py --dataset lastfm-2k --regs [1e-2] --lr 0.001 --k 5
```



Last Update Date: February 11, 2022.
