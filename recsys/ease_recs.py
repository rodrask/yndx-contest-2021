from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np

def to_BB(cooc_m, l2):
    return cooc_m + l2 * sparse.eye(cooc_m.shape[0])

def to_solution(BB, apply_fn=None):
    if apply_fn:
        inv_BB = np.linalg.inv(apply_fn(BB.todense())).astype(np.float32)
    else:
        inv_BB = np.linalg.inv(BB.todense()).astype(np.float32)
    inv_BB /= np.diag(inv_BB)
    return np.asarray(np.eye(inv_BB.shape[0]) - inv_BB).astype(np.float16)
