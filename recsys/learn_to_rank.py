import catboost
from catboost.core import CatBoost, Pool
import pandas as pd
import numpy as np
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer

from score_functions import recall

TrainData = namedtuple("TrainData", ["pairs", "groups", "labels","org_ids"])
TestData = namedtuple("TestData", ["pairs", "groups", "org_ids"])

class TfidfTransformer:
    def __init__(self, ints_column) -> None:
        self.transformer = TfidfVectorizer(lowercase=False)
        self.transformer.fit(ints_column.apply(lambda x: " ".join([str(i) for i in x])))
        self.columns = self.transformer.get_feature_names_out().tolist()

    def apply(self, column):
        m = self.transformer.transform(column.apply(lambda x: " ".join([str(i) for i in x])),).mean(axis=0)
        return np.asarray(m).reshape(-1).tolist(), self.columns

def user_f_index():
    user_f_index = [
        "n_reviews","n_plus_reviews","n_minus_reviews","n_travel_reviews",
        "mean_review_rating","mean_org_rating",
        "mean_plus_rating","mean_minus_rating","mean_travel_rating",
        "mean_bill","mean_plus_bill","mean_minus_bill","mean_travel_bill"
        ]
    return [f"user_{f}" for f in user_f_index]

def column_2_unique_str(columns):
    result = set()
    for c_list in columns:
        for item in c_list:
            result.add(str(item))
    return " ".join(result)

def _user_aggregate(review_group, aspects_transformer):
    num_columns_data=[
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
    np.mean(review_group[review_group.travel==1].average_bill)]
    text_columns_data, text_index = aspects_transformer.apply(review_group.aspects)
    return pd.Series(num_columns_data+text_columns_data, index=user_f_index()+text_index)

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

def _org_aggregate(review_group, feature_transformer):
    first = review_group.iloc[0]
    num_columns_data = [
    first.org_city,
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
    np.mean(review_group[review_group.ts>800].rating)]
    text_columns_data, text_index = feature_transformer.apply(review_group.combined_id)
    return pd.Series(num_columns_data+text_columns_data, index=orgs_f_index()+text_index)

class PoolBuilder:
    def __init__(self, orgs_df, reviews) -> None:
        self.orgs_df = orgs_df
        self.reviews = reviews
        self.aspect_transformer = TfidfTransformer(reviews.aspects)
        self.feature_transformer = TfidfTransformer(orgs_df.combined_id)

    def attach_orgs_df(self, selected_reviews):
        return selected_reviews.merge(self.orgs_df.drop(columns='city'), on='org_id',suffixes=("","_org"))
    
    def attach_user_vector(self, pairs):
        users = pairs.drop_duplicates('user_id')['user_id']
        users_reviews = self.reviews.merge(users, on='user_id')
        users_reviews = self.attach_orgs_df(users_reviews)

        users_reviews = users_reviews.groupby('user_id')\
            .apply(_user_aggregate, self.aspect_transformer).reset_index().fillna(0)
        pairs = pairs.merge(users_reviews, on='user_id')
        return pairs
    
    def attach_org_vector(self, pairs):
        orgs = pairs.drop_duplicates('org_id')['org_id']
        orgs_reviews = self.reviews.merge(orgs, on='org_id')
        orgs_reviews = self.attach_orgs_df(orgs_reviews)

        orgs_reviews = orgs_reviews.groupby('org_id').apply(_org_aggregate, self.feature_transformer).reset_index().fillna(0)
        pairs = pairs.merge(orgs_reviews, on='org_id')
        return pairs

    def combine_columns(pairs):
        pass
    
    def build_test_pool(self, test_df):
        test_pairs = test_df[['user_id','city','target']]\
            .explode('target')\
            .rename(columns={'target':'org_id',"city": "user_city"})
        test_pairs = self.attach_user_vector(test_pairs)
        test_pairs = self.attach_org_vector(test_pairs)
        test_pairs.sort_values(by="user_id", inplace=True)
        groups = test_pairs["user_id"].to_numpy()
        org_ids = test_pairs["org_id"]
        test_pairs.drop(columns=["user_id", "org_id"], inplace=True)
        return TestData(test_pairs, groups, org_ids)

    def build_big_pool(self, train_df, true_df):
            train_pairs = train_df[['user_id','city','target']]\
                .explode('target')\
                .rename(columns={'target':'org_id'})\
                .dropna()
            train_pairs['label'] = 0
            true_pairs = true_df[['user_id','city','target']]\
                .explode('target')\
                .rename(columns={'target':'org_id'})
            true_pairs['label'] = 1
            all_pairs = pd.concat([train_pairs, true_pairs],ignore_index=True)
            all_pairs.drop_duplicates(subset=['user_id','org_id'],keep='last', inplace=True)
            all_pairs.rename(columns={"city": "user_city"}, inplace=True)
            all_pairs = self.attach_user_vector(all_pairs)
            all_pairs = self.attach_org_vector(all_pairs)
            # self.combine_columns(all_pairs)
            all_pairs.sort_values(by="user_id", inplace=True)
            groups = all_pairs["user_id"].to_numpy()
            label = all_pairs.label.to_numpy()
            org_ids = all_pairs["org_id"]
            all_pairs.drop(columns=["user_id", "org_id", "label"], inplace=True)
            return TrainData(all_pairs, groups, label, org_ids)

    def build_pool(self, train_df, true_df):
        train_recall = recall(true_df, train_df)
        true_pairs = true_df[['user_id','target']]\
            .explode('target')\
            .rename(columns={'target':'org_id'})

        labels_one = set([(t.user_id, t.org_id) for t in true_pairs.itertuples()])
        train_pairs = train_df[train_recall>0][['user_id','city','target']]\
            .explode('target')\
            .rename(columns={'target':'org_id', "city": "user_city"})
        train_pairs['label'] = train_pairs.apply(lambda r: (r.user_id, r.org_id) in labels_one, axis=1)
        train_pairs['label'] = train_pairs['label'].astype(int)

        train_pairs = self.attach_user_vector(train_pairs)
        train_pairs = self.attach_org_vector(train_pairs)
        # self.combine_columns(all_pairs)
        train_pairs.sort_values(by="user_id", inplace=True)
        groups = train_pairs["user_id"].to_numpy()
        label = train_pairs.label.to_numpy()
        org_ids = train_pairs["org_id"]
        train_pairs.drop(columns=["user_id", "org_id", "label"], inplace=True)
        return TrainData(train_pairs, groups, label, org_ids)

def train_data_to_pool(train_data:TrainData):
    return Pool(data=train_data.pairs,
                  group_id=train_data.groups,
                  label=train_data.labels,
                  feature_names=train_data.pairs.columns.values.tolist(),
                  cat_features=["user_city","org_city"])

def test_data_to_pool(test_data:TestData):
    return Pool(data=test_data.pairs,
                  group_id=test_data.groups,
                  feature_names=test_data.pairs.columns.values.tolist(),
                  cat_features=["user_city","org_city"])


def split_4_ranking(train_data:TrainData, test_frac=0.25):
    pairs, groups, labels, org_ids = train_data
    users = np.unique(groups)
    test_users = np.random.choice(users, int(len(users) * test_frac))
    mask = np.in1d(groups, test_users)
    return TrainData(pairs[~mask], groups[~mask], labels[~mask], org_ids[~mask]), \
            TrainData(pairs[mask], groups[mask], labels[mask], org_ids[mask])

def cb_predict(model, pool, user_ids, org_ids):
    pairs = org_ids.to_frame().rename(columns={"org_id":"target"})
    pairs['user_id'] = user_ids
    pairs['score'] = model.predict(pool)
    return pairs.sort_values(by='score',ascending=False)\
    .groupby('user_id')[['target']].agg(lambda x: list(x)[:20]).reset_index()
