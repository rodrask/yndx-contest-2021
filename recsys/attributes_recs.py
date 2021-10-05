from sklearn.preprocessing import LabelEncoder
import numpy as np
from i2i_functions import Encoders
from scipy import sparse

class AttrEncoders:
    def __init__(self, orgs, features) -> None:
        self.feature_enc = LabelEncoder()
        self.feature_enc.fit(features.feature_id)
        self.idf_weight = np.zeros_like(self.feature_enc.classes_, dtype=float)
        idxs = self.feature_enc.transform(features.feature_id)
        self.idf_weight[idxs] = np.log(1/features.count_normed.to_numpy())

        self.org_enc = LabelEncoder()
        self.org_enc.fit(orgs['org_id'])
        exploded = orgs[orgs.features_l>0][['org_id','features_id']].explode('features_id')
        
        orgs_idx = self.org_enc.transform(exploded['org_id'])
        features_idx = self.feature_enc.transform(exploded['features_id'])
        data = np.ones_like(orgs_idx)
        self.features_mat = sparse.coo_matrix((data, (orgs_idx, features_idx))).tocsr()
        
    
    def build_feature_org_mat(self, i2i_mat, encoders:Encoders):
        idxs = range(i2i_mat.shape[0])
        org_ids = encoders.decode_orgs(idxs)
        org_idxs = self.org_enc.transform(org_ids)
        features = self.features_mat[org_idxs,:]
        self.feature_2_item = np.matmul(i2i_mat, features.toarray()).T
    
    def _get_features(self, org_ids):
        org_idxs = self.org_enc.transform(org_ids)
        return  self.features_mat[org_idxs,:].sum(axis=0) * self.idf_weight

    def predict(self, city, org_ids):
        feature_vec = self._get_features(org_ids)
