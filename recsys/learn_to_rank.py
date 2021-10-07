import catboost
from catboost.core import CatBoost, Pool
import pandas as pd
import numpy as np

def user_f_index():
    user_f_index = [
        "n_reviews","n_plus_reviews","n_minus_reviews","n_travel_reviews",
        "mean_review_rating","mean_org_rating",
        "mean_plus_rating","mean_minus_rating","mean_travel_rating",
        "mean_bill","mean_plus_bill","mean_minus_bill","mean_travel_bill",
        ]
    return [f"user_{f}" for f in user_f_index]

def _user_aggregate(review_group):
    return pd.Series(data=[
    len(review_group),
    len(review_group[review_group.good>0]),
    len(review_group[review_group.good==0]),
    len(review_group[review_group.travel==1]),
    
    np.mean(review_group.rating),
    np.mean(review_group.rating_org),
    np.mean(review_group[review_group.good>0].rating_org),
    np.mean(review_group[review_group.good==0].rating_org),
    np.mean(review_group[review_group.travel==1].rating_org),

    np.mean(review_group.average_bill),
    np.mean(review_group[review_group.good>0].average_bill),
    np.mean(review_group[review_group.good==0].average_bill),
    np.mean(review_group[review_group.travel==1].average_bill)],index = user_f_index())

def orgs_f_index():
    orgs_f_index=[
        "city",
        "mean_bill",
        "rating",
        "rubrics_len",
        "features_len",
        
        "n_org_reviews",
        "n_fresh_reviews",
        "n_plus_reviews",
        "n_minus_reviews",
        "n_travel_reviews",

        "mean_review_rating",
        "fresh_review_rating"]
    return [f"org_{f}" for f in orgs_f_index]

def _org_aggregate(review_group):
    first = review_group.iloc[0]
    return pd.Series(data=[
    int(first.org_city=="msk"),
    first.average_bill,
    first.rating_org,
    len(first.rubrics_id),
    len(first.features_id),
    
    len(review_group),
    len(review_group[review_group.ts>800]),
    len(review_group[review_group.good>0]),
    len(review_group[review_group.good==0]),
    len(review_group[review_group.travel==1]),
    
    np.mean(review_group.rating),
    np.mean(review_group[review_group.ts>800].rating)],index = orgs_f_index())

class PoolBuilder:
    def __init__(self, orgs_df, reviews) -> None:
        self.orgs_df = orgs_df
        self.reviews = reviews

    def attach_orgs_df(self, selected_reviews):
        return selected_reviews.merge(self.orgs_df.drop(columns='city'), on='org_id',suffixes=("","_org"))
    
    def attach_user_vector(self, pairs):
        users = pairs.drop_duplicates('user_id')['user_id']
        users_reviews = self.reviews.merge(users, on='user_id')
        users_reviews = self.attach_orgs_df(users_reviews)

        users_reviews = users_reviews.groupby('user_id').apply(_user_aggregate).reset_index().fillna(0)
        pairs = pairs.merge(users_reviews, on='user_id')
        return pairs
    
    def attach_org_vector(self, pairs):
        orgs = pairs.drop_duplicates('org_id')['org_id']
        orgs_reviews = self.reviews.merge(orgs, on='org_id')
        orgs_reviews = self.attach_orgs_df(orgs_reviews)

        orgs_reviews = orgs_reviews.groupby('org_id').apply(_org_aggregate).reset_index().fillna(0)
        pairs = pairs.merge(orgs_reviews, on='org_id')
        return pairs

    def combine_columns(pairs):
        pass


    def build_pool(self, train_df, true_df=None):
        train_mode = true_df is not None
        train_pairs = train_df[['user_id','city','target']]\
            .explode('target')\
            .rename(columns={'target':'org_id'})\
            .dropna()
        if train_mode:
            train_pairs['label'] = 0
            true_pairs = true_df[['user_id','city','target']]\
                .explode('target')\
                .rename(columns={'target':'org_id'})
            true_pairs['label'] = 1
            all_pairs = pd.concat([train_pairs, true_pairs],ignore_index=True)
            all_pairs.drop_duplicates(subset=['user_id','org_id'],keep='last', inplace=True)
        all_pairs["user_city"] = (all_pairs["city"] == "msk").astype(int)
        all_pairs = self.attach_user_vector(all_pairs)
        all_pairs = self.attach_org_vector(all_pairs)
        # self.combine_columns(all_pairs)
        all_pairs.sort_values(by="user_id", inplace=True)
        groups = all_pairs["user_id"].to_numpy()
        label = all_pairs.label.to_numpy()
        all_pairs.drop(columns=["user_id","org_id","label","city"], inplace=True)
        return all_pairs, groups, label


def train_model(pool, params):
    model = catboost.CatBoost(params)
    model.fit(pool)

def rank_items(model, pool):
    pass
