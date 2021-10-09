from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np
import pandas as pd

def city_mask(orgs_df, orgs_encoder, city):
    idxs = orgs_encoder.transform(orgs_df[orgs_df.city==city]['org_id'])
    mask = np.zeros_like(orgs_encoder.classes_)
    mask[idxs] = 1
    return mask

def msk_mask(orgs_df, orgs_encoder):
    return city_mask(orgs_df, orgs_encoder, 'msk')

def spb_mask(orgs_df, orgs_encoder):
    return city_mask(orgs_df, orgs_encoder, 'spb')

def aspects_matrix(reviews, users_encoder, aspects_encoder):
    exploded = reviews[reviews.aspects_l>0][['user_id','aspects']].explode('aspects')
    users_idx = users_encoder.transform(exploded['user_id'])
    aspects_idx = aspects_encoder.transform(exploded['aspects'])
    data = np.ones_like(users_idx)
    return sparse.coo_matrix((data, (users_idx, aspects_idx))).tocsr()

def features_matrix(orgs, orgs_encoder, features_encoder):
    exploded = orgs[orgs.features_l>0][['org_id','features_id']].explode('features_id')
    orgs_idx = orgs_encoder.transform(exploded['org_id'])
    features_idx = features_encoder.transform(exploded['features_id'])
    data = np.ones_like(orgs_idx)
    return sparse.coo_matrix((data, (orgs_idx, features_idx))).tocsr()

def rubric_matrix(orgs, orgs_encoder, rubrics_encoder):
    exploded = orgs[orgs.rubrics_l>0][['org_id','rubrics_id']].explode('rubrics_id')
    orgs_idx = orgs_encoder.transform(exploded['org_id'])
    rubrics_idx = rubrics_encoder.transform(exploded['rubrics_id'])
    data = np.ones_like(orgs_idx)
    return sparse.coo_matrix((data, (orgs_idx, rubrics_idx))).tocsr()

#user_id 	city 	in_test 	n_reviews
def train_test_split(reviews, min_ts):
    test_reviews = reviews[
        (reviews.ts >= min_ts) & \
        (reviews.good > 0) & (reviews.travel > 0)]
        
    test_users = test_reviews.user_id.unique()
    
    train_reviews = reviews[(reviews.good > 0) & \
                            (reviews.ts <= min_ts) & \
                            reviews.user_id.isin(test_users)]
    train_reviews = train_reviews.groupby(['user_id','user_city'])['org_id']\
        .aggregate(list).reset_index().rename(columns={'user_city':'city'})
    
    test_users = train_reviews.user_id.unique()

    test_reviews = test_reviews[test_reviews.user_id.isin(test_users)][['user_id','org_id']]\
        .groupby('user_id')['org_id']\
        .aggregate(list).reset_index().rename(columns={'org_id':"target"})

    return train_reviews, test_reviews