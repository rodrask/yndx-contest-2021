
from _typeshed import Self
import catboost
from catboost.core import CatBoost, Pool


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
        pool = Pool()
        def _apply(row):
            user_vector = self.get_user_vector(row['user_id'])
            for org_id in row['preds']:
                org_vector = self.get_org_vector(org_id)
                if is_test:
                    target = int(org_id in row['true_preds'])
                    result = self.combine_vecs(user_vector, org_vector, target)
                else:
                    result = self.combine_vecs(user_vector, org_vector)
                return result
        return Pool(train_df.parallel_apply(_apply, axis=1))

def train_model(pool, params):
    model = catboost.CatBoost(params)
    model.fit(pool)

def rank_items(model, pool):
    
