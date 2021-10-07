
from _typeshed import Self
import catboost
from catboost.core import CatBoost, Pool
import pandas as pd

class PoolBuilder:
    def __init__(self, users_df, orgs_df) -> None:
        self.user_features = users_df[['user_id','city','n_reviews','mean_score','mean_aspects','n_travels']].set_index("user_id")
        self.org_features = orgs_df[['org_id','city',
                                    'average_bill','rating','rubrics_l',
                                    'features_l','n_reviews',
                                    'mean_score','mean_aspects',
                                    'n_travels']].set_index("org_id")
    
    def get_user_vector(self, user_id):
        return self.user_features.loc[user_id]
    
    def get_org_vector(self, org_id):
        return self.org_features.loc[org_id]

    def combine_vecs(use_vector, org_vector, target=None):
        pass


    def build_pool(self, train_df, is_test=False):
        true_pairs = train_df[['user_id','true_preds']]\
            .explode('true_preds')\
            .rename(columns={'true_preds':'org_id'})
        true_pairs['label'] = 1
        rest_pairs = train_df[['user_id','preds']]\
            .explode('preds')\
            .rename(columns={'preds':'org_id'})
        true_pairs['label'] = 0
        all_pairs = pd.concat([true_pairs, rest_pairs],ignore_index=True)
        all_pairs.drop_duplicates(subset=['user_id','org_id'], inplace=True)
        self.attach_user_vector(all_pairs)
        self.attach_org_vector(all_pairs)
        self.combine_columns(all_pairs)
        return Pool(all_pairs,label="label",group_id="user_id")


def train_model(pool, params):
    model = catboost.CatBoost(params)
    model.fit(pool)

def rank_items(model, pool):
    
