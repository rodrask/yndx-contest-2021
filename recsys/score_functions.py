import pandas as pd
import numpy as np
from pandarallel import pandarallel

pd.options.mode.chained_assignment = None  # default='warn'

# Df format: user_id [org_ids,...] 
# Columns: user_id, target
def mnap(y_true, y_predict, N=20):
    match = y_true.merge(y_predict, on='user_id')

    def _user_mnap(y_true, preds, N):
        preds = preds[:N]
        scores = np.zeros_like(preds)
        weights = np.array([1/idx for idx in range(1, len(preds)+1)])
        for idx, pred in enumerate(preds, 1):
            if pred in y_true:
                scores[idx-1] = 1 
        return np.sum(np.cumsum(scores)*scores / weights) / min(len(y_true), N)
    return match.parallel_apply(lambda r: _user_mnap(r['target_x'],r['target_y'], N), axis=1).mean()

def recall(y_true, y_predict,N=20):
    match = y_true.merge(y_predict, on='user_id')
    def _user_recall(y_true, preds):
        return len(np.intersect1d(y_true, preds)) / (0.0001+len(y_true))
    return match.parallel_apply(lambda r: _user_recall(r['target_x'],r['target_y'][:N]), axis=1).mean()

def compare_ranks(y_true, y_predict_x, y_predict_y, cols):
    match = y_true.merge(y_predict_x, on='user_id', suffixes=('_true',''))
    match = match.merge(y_predict_y, on='user_id', suffixes=('_x','_y'))
    def pair_intersection(y_true, preds):
        return len(np.intersect1d(y_true, preds))
    def triple_union(y_true, preds_x, preds_y):
        inter_x = np.intersect1d(y_true, preds_x)
        inter_y = np.intersect1d(y_true, preds_y)
        return len(np.union1d(inter_x, inter_y))
    def _apply(row):
        row[cols[0]] = pair_intersection(row['target_true'],row['target_x'])
        row[cols[1]] = pair_intersection(row['target_true'],row['target_y'])
        row[f'{cols[0]}_{cols[1]}'] = triple_union(row['target_true'], row['target_x'], row['target_y'])
        return row

    result = match.parallel_apply(_apply, axis=1)[[cols[0],cols[1],f'{cols[0]}_{cols[1]}']]
    result[f'imp_{cols[0]}'] = result[f'{cols[0]}_{cols[1]}'] - result[cols[1]]
    result[f'imp_{cols[1]}'] = result[f'{cols[0]}_{cols[1]}'] - result[cols[0]]
    return result

def print_score(score):
    print(f'MNAP-score: {100*score:.2f}')


def fallback_with_top_recs(test_reviews, orgs, N=20):

    top_spb = orgs[(orgs.city=='spb')&(orgs.rating>4.8)].sort_values(by='n_reviews', ascending=False)[:N]['org_id'].to_numpy()
    top_msk = orgs[(orgs.city=='msk')&(orgs.rating>4.8)].sort_values(by='n_reviews', ascending=False)[:N]['org_id'].to_numpy()
    top_dict={
        'msk':top_spb,
        'spb':top_msk
    }
    result = test_reviews[['user_id']]
    result['target'] = test_reviews.apply(lambda r: top_dict[r['city']] if len(r['target'])== 0 else r['target'], axis=1)
    return result

def save_predictions(preds, path='answers.csv'):
    preds['target_s'] = preds['target'].apply(lambda arr: ' '.join(arr))
    preds.to_csv(path, columns=['user_id','target_s'], header=['user_id','target'], index=None, sep=',')


def validate_preds(preds, orgs_df, users_df, N=20):
    org2_city = {r.org_id:r.city for r in orgs_df.itertuples()}
    user2_city = {r.user_id:r.city for r in users_df.itertuples()}
    assert len(preds) == sum(users_df.in_test)
    for row in preds.itertuples():
        user_city = user2_city[row.user_id]
        orgs_cities = [org2_city[o] for o in row.target]
        assert len(orgs_cities) <= N
        assert user_city not in orgs_cities, f"{row.user_id} {user_city} {row.target} {orgs_cities}"
    print("All good")


def combine_preds(train_df, ease, attrs, min_len=4):
    preds_combined = pd.merge(ease, attrs, on='user_id',suffixes=('_e','_a'))
    preds_combined = preds_combined.merge(train_df[['user_id','org_id']], on='user_id')
    def _apply(row):
        index = ['user_id', 'city', 'target']
        if len(row['org_id']) <= min_len:
            return pd.Series(data=(row['user_id'], row['city_a'], row['target_a']), index=index)
        else:
            return pd.Series(data=(row['user_id'], row['city_e'], row['target_e']), index=index)
    return preds_combined.apply(_apply, axis=1)
