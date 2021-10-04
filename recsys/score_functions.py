import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

# Df format: user_id [org_ids,...] 
# Columns: user_id, target
def mnap(y_true, y_predict, N=20):
    match = y_true.merge(y_predict, on='user_id')
    return match.apply(lambda r: user_mnap(r['target_x'],r['target_y'], N), axis=1).mean()

def recall(y_true, y_predict):
    match = y_true.merge(y_predict, on='user_id')
    return match.apply(lambda r: user_recall(r['target_x'],r['target_y']), axis=1).mean()

def user_recall(y_true, preds):
    return len(np.intersect1d(y_true, preds)) / len(y_true)
         
def user_mnap(y_true, preds, N):
    preds = preds[:N]
    scores = np.zeros_like(preds)
    weights = np.array([1/idx for idx in range(1, len(preds)+1)])
    for idx, pred in enumerate(preds, 1):
        if pred in y_true:
            scores[idx-1] = 1 
    return np.sum(np.cumsum(scores)*scores / weights) / min(len(y_true), N)

def print_score(score):
    print(f'MNAP-score: {100*score:.2f}')


def fallback_with_top_recs(test_reviews, orgs, N=20):
    top_spb = orgs[(orgs.city=='spb')&(orgs.mean_score>4.8)].sort_values(by='n_reviews', ascending=False)[:N]['org_id'].to_numpy()
    top_msk = orgs[(orgs.city=='msk')&(orgs.mean_score>4.8)].sort_values(by='n_reviews', ascending=False)[:N]['org_id'].to_numpy()
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
