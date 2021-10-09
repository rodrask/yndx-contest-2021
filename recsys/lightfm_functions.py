from typing import Counter, NamedTuple
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np
import pandas as pd
from transform_functions import *
from collections import Counter
from pandarallel import pandarallel


def prepare_reviews_lightfm()