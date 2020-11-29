from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from src.constants import current_date
import numpy as np
import pickle
import matplotlib as plt
import seaborn as sns


def eval_figures_model(model, df, y):
    fig, ax = plt.subplots(nrows=2, figsize=(15, 15))
    preds = model.predict(df)
    sns.scatterplot(x=preds, y=y, ax=ax[0])
    sns.residplot(preds, y, ax=ax[1])


def train_lr(train_df, train_y, valid_df, valid_y):
    model = LinearRegression()
    model.fit(train_df, train_y)
    pred = model.predict(valid_df)
    mse_res, msle_res = np.sqrt(mean_squared_error(pred, valid_y.values)), np.sqrt(
        mean_squared_log_error(pred, valid_y.values))
    print("For Linear Regression")
    print("MSE: {}".format(mse_res))
    print("MSLE: {}".format(msle_res))
    with open("models/lr-{}.h5".format(current_date), "wr") as file:
        pickle.dump(model, file)
    eval_figures_model(model, train_df, train_y)
    eval_figures_model(model, valid_df, valid_y)

    return model


def train_ridge(train_df, train_y, valid_df, valid_y):
    model = Ridge()
    model.fit(train_df, train_y)
    pred = model.predict(valid_df)
    mse_res, msle_res = np.sqrt(mean_squared_error(pred, valid_y.values)), np.sqrt(
        mean_squared_log_error(pred, valid_y.values))
    print("For Ridge Regression")
    print("MSE: {}".format(mse_res))
    print("MSLE: {}".format(msle_res))
    with open("models/ridge-{}.h5".format(current_date), "wr") as file:
        pickle.dump(model, file)
    eval_figures_model(model, train_df, train_y)
    eval_figures_model(model, valid_df, valid_y)
    return model


def train_lasso(train_df, train_y, valid_df, valid_y):
    model = Lasso(alpha=1)
    model.fit(train_df, train_y)
    pred = model.predict(valid_df)
    mse_res, msle_res = np.sqrt(mean_squared_error(pred, valid_y.values)), np.sqrt(
        mean_squared_log_error(pred, valid_y.values))
    print("For Lasso Regression")
    print("MSE: {}".format(mse_res))
    print("MSLE: {}".format(msle_res))
    with open("models/lasso-{}.h5".format(current_date), "wr") as file:
        pickle.dump(model, file)
    eval_figures_model(model, train_df, train_y)
    eval_figures_model(model, valid_df, valid_y)
    return model
