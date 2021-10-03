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

def top_recs(orgs, test_reviews, N=20):
    top_spb = orgs[(orgs.city=='spb')&(orgs.mean_score>4.8)].sort_values(by='n_reviews', ascending=False)[:N]['org_id'].to_numpy()
    top_msk = orgs[(orgs.city=='msk')&(orgs.mean_score>4.8)].sort_values(by='n_reviews', ascending=False)[:N]['org_id'].to_numpy()

    result = test_reviews[['user_id']]
    result['target'] = test_reviews.apply(lambda r: top_msk if r['city']== 'spb' else top_spb, axis=1)
    return result

def save_predictions(preds, path='answers.csv'):
    preds['target_s'] = preds['target'].apply(lambda arr: ' '.join(arr))
    preds.to_csv(path, columns=['user_id','target_s'], header=['user_id','target'], index=None, sep=',')
