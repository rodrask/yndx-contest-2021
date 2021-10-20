import numpy as np
import pandas as pd
from transform_functions import *
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer

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

def attach_user_vector(self, pairs):
    users = pairs.drop_duplicates('user_id')['user_id']
    users_reviews = self.reviews.merge(users, on='user_id')
    users_reviews = self.attach_orgs_df(users_reviews)

    pairs = pairs.merge(users_reviews, on='user_id')
    return pairs

def attach_org_vector(self, pairs):
    orgs = pairs.drop_duplicates('org_id')['org_id']
    orgs_reviews = self.reviews.merge(orgs, on='org_id')
    orgs_reviews = self.attach_orgs_df(orgs_reviews)

    orgs_reviews = orgs_reviews
    pairs = pairs.merge(orgs_reviews, on='org_id')
    return pairs


def featurize_items(orgs, reviews):
    aspect_transformer = TfidfTransformer(reviews.aspects)
    feature_transformer = TfidfTransformer(orgs.combined_id)
    df = reviews.merge(orgs.drop(columns='city'), on='org_id',suffixes=("","_org"))
    users_features = df.groupby('user_id')\
        .apply(_user_aggregate, aspect_transformer).reset_index().fillna(0)
    org_features = df.groupby('org_id').apply(_org_aggregate, feature_transformer).reset_index().fillna(0)
    return users_features, org_features
