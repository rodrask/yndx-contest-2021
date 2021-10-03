from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np

def prepare_items_4(reviews, users)

def ease_solution(X, l2):
    CC_mat = X.T * X
    CC_mat += l2 * sparse.eye(CC_mat.shape[0])
    CC_mat = sparse.linalg.inv(CC_mat)
    CC_mat /= np.diag(CC_mat)
    return np.asarray(np.eye(CC_mat.shape[0]) - CC_mat)


#test_users: user_id, [org_ids]
def i2i_predict(i2i_mat, test_users, city_masks, user_enc, org_enc, N=20):
    for row in test_users.iteritems():
        pass