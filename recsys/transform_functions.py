from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np

def index_items(items):
    encoder = LabelEncoder()
    encoder.fit(items)
    return encoder

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

#user_id 	city 	in_test 	n_reviews 	mean_score 	mean_aspects 	n_travels
def train_test_split(reviews, users, min_ts=500, test_share=0.2):
    potential_users = users[(~users.in_test) & (users.n_travels>0) & (users.n_reviews>5)][['user_id']]
    potential_reviews = reviews[(reviews.rating>=4.0)&(reviews.travel>0)&(reviews.ts>min_ts)]
    potential_reviews = potential_reviews.merge(potential_users, on='user_id')

