from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter


def load_csv(path, **kwargs):
    return pd.read_csv(path, sep=',', header=0, **kwargs, low_memory=False)


def load_users(path='users.csv'):
    return load_csv(path, dtype={'user_id':str, 'city':str})

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
    orgs_df['average_bill'].astype(int)
    orgs_df['rubrics_l'] = orgs_df['rubrics_id'].apply(len)
    orgs_df['features_l'] = orgs_df['features_id'].apply(len)
    return orgs_df

#user_id,org_id,rating,ts,aspects
def load_reviews(path='reviews.csv', users_df=None, orgs_df=None):
    reviews_df = load_csv(path,  
                       dtype={'user_id':str, 'org_id':str, 'rating':float,'ts':int},
                       converters={'aspects': parse_list_ints})
    reviews_df['aspects_l']=reviews_df['aspects'].apply(len)
    
    if users_df is not None:
        reviews_df = reviews_df.merge(users_df[['user_id','city']],on='user_id')
        reviews_df.rename({'city': 'user_city'}, axis=1, inplace=True)
    if orgs_df is not None:
        reviews_df = reviews_df.merge(orgs_df[['org_id','city']],on='org_id')
        reviews_df.rename({'city': 'org_city'}, axis=1, inplace=True)
    if 'org_city' in reviews_df and 'user_city' in reviews_df:
        reviews_df['travel'] = (reviews_df['org_city'] != reviews_df['user_city']).astype(int)
    return reviews_df

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
