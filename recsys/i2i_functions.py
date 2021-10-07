from typing import Counter, NamedTuple
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np
import pandas as pd
from transform_functions import *
from collections import Counter
from pandarallel import pandarallel

class Encoders:
    def __init__(self, reviews) -> None:
        self.users_enc = LabelEncoder()
        users_items = reviews['user_id'].drop_duplicates()
        self.in_users = set(users_items)
        self.users_enc.fit(users_items)

        self.orgs_enc = LabelEncoder()
        orgs_items = reviews[['org_id','org_city']].drop_duplicates()
        self.in_orgs = set(orgs_items['org_id'])
        self.orgs_enc.fit(orgs_items['org_id'])

        msk_idxs = self.orgs_enc.transform(orgs_items[orgs_items['org_city'] == 'msk']['org_id'])
        msk_mask = np.zeros_like(self.orgs_enc.classes_, dtype=int) 
        msk_mask[msk_idxs] = 1
        
        spb_idxs = self.orgs_enc.transform(orgs_items[orgs_items['org_city'] == 'spb']['org_id'])
        spb_mask = np.zeros_like(self.orgs_enc.classes_, dtype=int) 
        spb_mask[spb_idxs] = 1

        self.pred_masks = {
            'spb':spb_mask,
            'msk':msk_mask
        }
    
    def encode_users(self, users):
        if isinstance(users, str):
            users = [users]
        users = [u for u in users if u in self.in_users]    
        if users:
            return self.users_enc.transform(users)
        return []
        
    def decode_users(self, users_idxs):
        return self.users_enc.inverse_transform(users_idxs)

    def encode_orgs(self, orgs):
        if isinstance(orgs, str):
            orgs = [orgs]
        orgs = [o for o in orgs if o in self.in_orgs]    
        if orgs:
            return self.orgs_enc.transform(orgs)
        return []
        
    def decode_orgs(self, orgs_idxs):
        return self.orgs_enc.inverse_transform(orgs_idxs)
        
def prepare_reviews_i2i(reviews, orgs, 
                        min_reviews_per_user, 
                        min_org_reviews,
                        min_travels_reviews,
                        min_org_score):
    orgs = orgs[orgs.rating>=min_org_score]['org_id']
    reviews = reviews[reviews.good>0].merge(orgs, on='org_id')

    user_agg = reviews.groupby("user_id").agg(
        user_reviews = pd.NamedAgg(column="org_id", aggfunc="count")).reset_index()
    user_agg = user_agg[user_agg.user_reviews>=min_reviews_per_user]['user_id']
    reviews = reviews.merge(user_agg, on='user_id')
    
    org_agg = reviews.groupby("org_id").agg(
        org_reviews = pd.NamedAgg(column="org_id", aggfunc="count"),
        org_travels = pd.NamedAgg(column="travel", aggfunc="sum")).reset_index()
    org_agg = org_agg[(org_agg.org_travels>=min_travels_reviews) & (org_agg.org_reviews>=min_org_reviews)]['org_id']
    reviews = reviews.merge(org_agg, on='org_id')
    return (reviews, Encoders(reviews))

def reviews_matrix(reviews, encoders):
    users_idx = encoders.encode_users(reviews['user_id'])
    org_idx = encoders.encode_orgs(reviews['org_id'])
    data = np.ones_like(users_idx)
    return sparse.coo_matrix((data, (users_idx, org_idx))).tocsr()

def filter_by_min_value(CC_mat, min_value):
    CC_mat = CC_mat.copy()
    nonzero_mask = np.array(CC_mat[CC_mat.nonzero()] < min_value)[0]
    rows = CC_mat.nonzero()[0][nonzero_mask]
    cols = CC_mat.nonzero()[1][nonzero_mask]
    CC_mat[rows, cols] = 0
    CC_mat.eliminate_zeros()
    return CC_mat

def CC_2_pmi(CC_mat, min_CC=10):
    CC_mat = filter_by_min_value(CC_mat, min_CC)
    marg_sum = np.asarray(CC_mat.sum(axis=1)).reshape(-1)
    log_total = np.log(marg_sum.sum())
    log_marg = np.log(0.01+marg_sum)
    substr = np.add.outer(log_marg, log_marg)
    CC_log = np.log(0.01+CC_mat.todense())
    return np.asarray(log_total + CC_log - substr)

def CC_2_J(CC_mat):
    marg_sum = np.asarray(CC_mat.sum(axis=0)).reshape(-1)
    denom = 0.001 + np.add.outer(marg_sum, marg_sum) - CC_mat
    result = CC_mat.todense() / denom
    return np.asarray(result)

def ease_solution(CC_mat, l2=0.01):
    CC_mat += l2 * sparse.eye(CC_mat.shape[0])
    CC_mat = np.linalg.inv(CC_mat.todense())
    CC_mat /= np.diag(CC_mat)
    return np.asarray(np.eye(CC_mat.shape[0]) - CC_mat)

#test_users: user_id, city, [org_id]
def i2i_predict(i2i_mat, test_users, encoders:Encoders, N=20):
    out_index = ['user_id','city','target','target_values']
    def _apply(row):
        history = row.org_id
        orgs_idxs = encoders.encode_orgs(history)
        if len(orgs_idxs) > 0:
            predicts = np.sum(i2i_mat[orgs_idxs,:], axis=0) - 1e6 * encoders.pred_masks[row.city]
            predicts[orgs_idxs] -= 1e6
            top_N = np.argpartition(-predicts, range(N))[:N]
            top_scores = [s for s in predicts[top_N] if s > 0]
            top_N = top_N[:len(top_scores)]
            data = (row.user_id, row.city, encoders.decode_orgs(top_N), top_scores)
        else:
            data = (row.user_id, row.city, [], [])
        return pd.Series(data, index=out_index)
    return test_users.parallel_apply(_apply, axis=1)


def merge_ranks(ranks, weights, N=20):
    final = Counter()
    for rank, weight in zip(ranks,weights):
        for pos,item in enumerate(rank, 1):
            final[item] += weight * 1/pos
    return [i[0] for i in  final.most_common(N)]


def i2i_predict_merged(i2i_mats, i2i_weights, test_users, encoders:Encoders, N=20):
    result = []
    for row in test_users.itertuples():
        history = row.org_id
        orgs_idxs = encoders.encode_orgs(history)
        if len(orgs_idxs) > 0:
            ranks = []
            for i2i in i2i_mats:
                predicts = np.sum(i2i[orgs_idxs,:], axis=0) - 1e6 * encoders.pred_masks[row.city]
                predicts[orgs_idxs] -= 1e6
                top_N = np.argpartition(-predicts, range(N))[:N]
                top_scores = [s for s in predicts[top_N] if s > 0]
                top_N = top_N[:len(top_scores)]
                ranks.append(top_N)
            merged = merge_ranks(ranks, i2i_weights, N)
            result.append((row.user_id, row.city, encoders.decode_orgs(merged)))
        else:
            result.append((row.user_id, row.city, []))
    return pd.DataFrame(result,columns=['user_id','city','target'])

