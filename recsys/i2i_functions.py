from typing import NamedTuple
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np
import pandas as pd
from transform_functions import *
from collections import namedtuple

class Encoders:
    def __init__(self, users_items, orgs_items) -> None:
        self.users_enc = LabelEncoder()
        self.in_users = set(users_items)
        self.users_enc.fit(users_items)

        self.orgs_enc = LabelEncoder()
        self.in_orgs = set(orgs_items['org_id'])
        self.orgs_enc.fit(orgs_items['org_id'])

        msk_idxs = self.orgs_enc.transform(orgs_items[orgs_items['city'] == 'msk']['org_id'])
        msk_mask = np.zeros_like(self.orgs_enc.classes_, dtype=int) 
        msk_mask[msk_idxs] = 1.0
        
        spb_idxs = self.orgs_enc.transform(orgs_items[orgs_items['city'] == 'spb']['org_id'])
        spb_mask = np.zeros_like(self.orgs_enc.classes_, dtype=int) 
        spb_mask[spb_idxs] = 1

        self.pred_masks = {
            'spb':msk_mask,
            'msk':spb_mask
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
        
def prepare_reviews_i2i(reviews, users, orgs, min_reviews_per_user=5, min_travels_reviews=2):
    users = users[users.n_reviews>=min_reviews_per_user]['user_id']
    orgs = orgs[orgs.n_travels>min_travels_reviews][['org_id','city']]
    reviews = reviews[reviews.rating>=4.0].merge(users, on='user_id').merge(orgs['org_id'], on='org_id')
    return (reviews, Encoders(users, orgs))

def reviews_matrix(reviews, encoders):
    users_idx = encoders.encode_users(reviews['user_id'])
    org_idx = encoders.encode_orgs(reviews['org_id'])
    data = np.ones_like(users_idx)
    return sparse.coo_matrix((data, (users_idx, org_idx))).tocsr()

def CC_2_pmi(CC_mat):
    marg_sum = np.asarray(CC_mat.sum(axis=1)).reshape(-1)
    log_total = np.log(marg_sum.sum())
    log_marg = np.log(0.01+marg_sum)
    substr = np.add.outer(log_marg, log_marg)
    CC_log = np.log(0.01+CC_mat.todense())
    return log_total + CC_log - substr

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
    result = []
    for row in test_users.itertuples():
        history = row.org_id
        orgs_idxs = encoders.encode_orgs(history)
        if len(orgs_idxs) > 0:
            predicts = np.sum(i2i_mat[orgs_idxs,:], axis=0)
            np.subtract(predicts,1000 * encoders.pred_masks[row.city], predicts)
            predicts[orgs_idxs] -= 1000
            top_N = np.argpartition(-predicts, range(N))[:N]
            result.append((row.user_id, encoders.decode_orgs(top_N), predicts[top_N]))
        else:
            result.append((row.user_id, [], []))
    return pd.DataFrame(result,columns=['user_id','target','target_values'])


