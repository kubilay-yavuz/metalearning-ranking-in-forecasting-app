from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack


def scale_features(df, non_scaled_features):
    df[df.columns.difference(non_scaled_features)] = pd.DataFrame(
        scale(df[df.columns.difference(non_scaled_features)]),
        columns=df[df.columns.difference(non_scaled_features)].columns)

    return df


def onehot_df(cons_temp, cat_features):
    linear_enc = OneHotEncoder()
    one_hot_encoded = linear_enc.fit_transform(cons_temp[cat_features]).toarray()
    train_df = hstack(
        [cons_temp[cons_temp.columns.difference(cat_features)].astype(float).drop(columns=["Consumption"]),
         one_hot_encoded])
    return train_df, linear_enc
