from lightfm.data import *
from lightfm import LightFM
import pandas as pd
import numpy as np


def build_dataset(user_features, orgs_features, reviews:pd.DataFrame):
	u_f_list = [c for c in user_features.columns if c != 'user_id']
	o_f_list = [c for c in orgs_features.columns if c != 'org_id']
	dataset = Dataset()

	dataset.fit(user_features['user_id'], orgs_features['org_id'], 
				user_features[u_f_list], orgs_features[o_f_list])

	(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN'])
                                                      for x in reviews.iterrows()))

