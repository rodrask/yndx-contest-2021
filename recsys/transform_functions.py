from sklearn.preprocessing import LabelEncoder

def index_items(items):
    encoder = LabelEncoder()
    encoder.fit(items)
    return encoder
     