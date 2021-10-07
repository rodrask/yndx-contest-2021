from sklearn.preprocessing import LabelEncoder
import numpy as np
from i2i_functions import Encoders
from scipy import sparse
import pandas as pd
from pandarallel import pandarallel


class AttrEncoders:
    def __init__(self, orgs, attrs_df, 
                 colname, 
                 org_colname) -> None:
        self.attr_enc = LabelEncoder()
        self.attr_enc.fit(attrs_df[colname])
        self.idf_weight = np.zeros_like(self.attr_enc.classes_, dtype=float)
        idxs = self.attr_enc.transform(attrs_df[colname])
        self.idf_weight[idxs] = np.log(1/attrs_df.count_normed.to_numpy())

        self.org_set = set(orgs['org_id'].unique())
        self.org_enc = LabelEncoder()
        self.org_enc.fit(orgs['org_id'])
        exploded = orgs[orgs[org_colname].str.len()>0][['org_id',org_colname]].explode(org_colname)
        
        orgs_idx = self.org_enc.transform(exploded['org_id'])
        attrs_idx = self.attr_enc.transform(exploded[org_colname])
        data = np.ones_like(orgs_idx)
        self.attrs_mat = sparse.coo_matrix((data, (orgs_idx, attrs_idx))).tocsr()
        

    def build_attr_org_mat(self, i2i_mat, encoders:Encoders):
        idxs = range(i2i_mat.shape[0])
        org_ids = encoders.decode_orgs(idxs)
        org_idxs = self.org_enc.transform(org_ids)
        self.attr_2_item = np.matmul(i2i_mat, self.attrs_mat[org_idxs,:].toarray()).T
    
    def _get_features(self, org_ids):
        org_idxs = self.org_enc.transform([o for o in org_ids if o in self.org_set])
        if len(org_idxs) > 0:
            return  np.asarray(self.attrs_mat[org_idxs,:].sum(axis=0)).reshape(-1) * self.idf_weight
        else:
            return None

    def attr_predict(self, test_users, encoders:Encoders, N=20):
        out_index = ['user_id','city','target','target_values']
        def _apply(row):
            history = row.org_id
            feature_vec = self._get_features(history)
            if feature_vec is not None:
                predicts = np.dot(feature_vec, self.attr_2_item) - 1e6 * encoders.pred_masks[row.city]
                orgs_idxs = encoders.encode_orgs(history)
                if len(orgs_idxs) > 0:
                    predicts[orgs_idxs] -= 1e6
                top_N = np.argpartition(-predicts, range(N))[:N]
                top_scores = [s for s in predicts[top_N] if s > 0]
                top_N = top_N[:len(top_scores)]
                data = (row.user_id, row.city, encoders.decode_orgs(top_N), top_scores)
            else:
                data = (row.user_id, row.city, [], [])
            return pd.Series(data, index=out_index)
        return test_users.parallel_apply(_apply, axis=1)


