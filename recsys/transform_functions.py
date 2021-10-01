from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np

def index_items(items):
    encoder = LabelEncoder()
    encoder.fit(items)
    return encoder

def reviews_to_matrix(reviews, users_encoder, orgs_encoder):
    users_idx = users_encoder.transform(reviews['user_id'])
    org_idx = orgs_encoder.transform(reviews['org_id'])
    data = np.ones_like(users_idx)
    return sparse.coo_matrix((data, (users_idx, org_idx)))

