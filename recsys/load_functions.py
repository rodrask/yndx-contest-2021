from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter


def load_csv(path, **kwargs):
    return pd.read_csv(path, sep=',', header=0, **kwargs, low_memory=False)

def load_users(path, test_users):
    result = load_csv(path, dtype={'user_id':str, 'city':str})
    test_ids = set(test_users['user_id'])
    result['in_test'] = result['user_id'].isin(test_ids).astype(int)
    return result

def parse_avg_bill(avg_bill:str) -> int:
    if avg_bill:
        value = int(float(avg_bill))
        if value > 100000:
            value /= 100
        elif value > 10000:
            value /= 10
        return value
    return -1

def parse_list_ints(items):
    if items:
        return [int(rubric) for rubric in items.split(' ')]
    return []

# org_id,city,average_bill,rating,rubrics_id,features_id
def load_orgs(path='organisations.csv'):
    orgs_df = load_csv(path, dtype={'org_id':str, 
                                   'city':str}, 
                      converters={'average_bill':parse_avg_bill, 
                                  'rubrics_id':parse_list_ints,
                                  'features_id':parse_list_ints})
    orgs_df['rating'].fillna(0, inplace=True)                             
    orgs_df['average_bill'].astype(int)
    orgs_df['combined_id'] = orgs_df.apply(lambda r: r['rubrics_id']+r['features_id'], axis=1)
    return orgs_df

#user_id,org_id,rating,ts,aspects
def load_reviews(path, users_df, orgs_df):
    reviews_df = load_csv(path,  
                       dtype={'user_id':str, 'org_id':str, 'rating':float,'ts':int},
                       converters={'aspects': parse_list_ints})
    reviews_df['rating'].fillna(0, inplace=True)
    reviews_df['good'] = (reviews_df.rating>=4.0).astype(int)

    reviews_df = reviews_df.merge(users_df[['user_id','city','in_test']], on='user_id')
    reviews_df.rename({'city': 'user_city'}, axis=1, inplace=True)
    reviews_df = reviews_df.merge(orgs_df[['org_id','city']],on='org_id')
    reviews_df.rename({'city': 'org_city'}, axis=1, inplace=True)
    reviews_df['travel'] = (reviews_df['org_city'] != reviews_df['user_city']).astype(int)
    
    return reviews_df
def enrich_orgs(orgs_df, reviews):
    n_reviews = reviews.groupby("org_id")['user_id'].agg(
        n_reviews='count')
    return orgs_df.merge(n_reviews, on="org_id")

def load_combined(paths, orgs_df):
    dfs = [load_csv(path=p, 
            names=['combined_id', 'combined_name'],
            dtype={'combined_id':int, 'combined_name':str}) for p in paths]
    result = pd.concat(dfs, ignore_index=True)
    result = pd.merge(result, items_popularity(orgs_df, 'combined_id'), 
                        left_on='combined_id', 
                        right_on='combined_id', 
                        how='left').sort_values(by='count', ascending=False)
    return result


#rubric_id,rubric_name
def load_rubrics(path='rubrics.csv', orgs_df=None):
    result =  load_csv(path=path, dtype={'rubric_id':int, 'rubric_name':str})
    if orgs_df is not None:
        result = pd.merge(result, items_popularity(orgs_df, 'rubrics_id'), 
                          left_on='rubric_id', 
                          right_on='rubrics_id', 
                          how='left').drop(columns='rubrics_id').sort_values(by='count', ascending=False)
    return result
    
    
#aspect_id,aspect_name
def load_aspects(path='aspects.csv'):
    result =  load_csv(path=path, dtype={'aspect_id':int, 'aspect_name':str})
    return result


#feature_id,feature_name
def load_features(path='features.csv', orgs_df=None):
    result = load_csv(path=path, dtype={'feature_id':int, 'feature_name':str})
    if orgs_df is not None:
        result = pd.merge(result, items_popularity(orgs_df, 'features_id'), 
                          left_on='feature_id', 
                          right_on='features_id', 
                          how='left').drop(columns='features_id').sort_values(by='count', ascending=False)
    return result

def items_popularity(orgs_df, colname):
    result = Counter()
    for items in orgs_df[colname].iteritems():
        result.update(items[1])
    result = pd.DataFrame(result.items(), columns=[colname, 'count'])
    result['count_normed'] = result['count'] / len(orgs_df)
    return result
