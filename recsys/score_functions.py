import pandas as pd
import numpy as np

# Df format: user_id [org_ids,...] 
# Columns: user_id, target
def mnap(y_true, y_predict, N=20):
    match = y_true.merge(y_predict, on='user_id')
    return match.apply(lambda r: user_mnap(r['target_x'],r['target_y'], N)).mean()

def recall(y_true, y_predict):
    match = y_true.merge(y_predict, on='user_id')
    return match.apply(lambda r: user_recall(r['target_x'],r['target_y'])).mean()

def user_recall(y_true, preds):
    return len(np.intersect1d(y_true, preds)) / len(y_true)
         
def user_mnap(y_true, preds, N):
    preds = preds[:N]
    scores = np.zeros_like(preds)
    weights = np.array([1/idx for idx in range(1, len(preds)+1)])
    for idx, pred in enumerate(preds, 1):
        if pred in y_true:
            scores[idx-1] = 1 
    return np.sum(np.cumsum(preds)*preds / weights) / min(len(y_true), N)
