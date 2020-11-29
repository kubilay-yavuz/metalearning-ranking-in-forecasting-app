import pandas as pd


def build_features():
    consumption_df = pd.read_csv("data/raw/RealTimeConsumption-01122014-09032020.csv", encoding='latin-1')
    consumption_df.columns = ["Date", "Hour", "Consumption"]
    consumption_df["Datetime"] = consumption_df["Date"] + " " + consumption_df["Hour"]
    consumption_df["Consumption"] = consumption_df["Consumption"].apply(lambda x: x.replace(",", "")).astype(float)
    consumption_df["Datetime"] = pd.to_datetime(consumption_df["Datetime"], format="%d.%m.%Y %H:%M")
    consumption_df = consumption_df.loc[consumption_df["Datetime"] < pd.to_datetime("2020-03-01 00:00:00")]

    seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
    month_to_season = dict(zip(range(1, 13), seasons))

    consumption_df["Hour"] = consumption_df["Hour"].apply(lambda x: x.split(":")[0]).astype(int)
    consumption_df["Day"] = consumption_df["Date"].apply(lambda x: x.split(".")[0]).astype(int)
    consumption_df["Month"] = consumption_df["Date"].apply(lambda x: x.split(".")[1]).astype(int)
    consumption_df["Year"] = consumption_df["Date"].apply(lambda x: x.split(".")[2]).astype(int)
    consumption_df["weekday"] = pd.to_datetime(consumption_df["Date"]).apply(lambda x: x.weekday()).astype(
        int)  # sonradan eklendi kaldırıp performans testin yab
    consumption_df["Season"] = consumption_df["Month"].map(month_to_season)  # sonradan sonradan eklendi perf test
    consumption_df["Quarter"] = pd.to_datetime(consumption_df["Date"]).dt.to_period("Q")
    # quarter ekle

    consumption_df['Hour_sin'] = np.sin(2 * np.pi * consumption_df['Hour'] / 23.0)
    consumption_df['Hour_cos'] = np.cos(2 * np.pi * consumption_df['Hour'] / 23.0)

    consumption_df['Day_sin'] = np.sin(2 * np.pi * consumption_df['Day'] / 31.0)
    consumption_df['Day_cos'] = np.cos(2 * np.pi * consumption_df['Day'] / 31.0)

    consumption_df['Month_sin'] = np.sin(2 * np.pi * consumption_df['Month'] / 12.0)
    consumption_df['Month_cos'] = np.cos(2 * np.pi * consumption_df['Month'] / 12.0)

    consumption_df['weekday_sin'] = np.sin(2 * np.pi * consumption_df['weekday'] / 6.0)
    consumption_df['weekday_cos'] = np.cos(2 * np.pi * consumption_df['weekday'] / 6.0)

    del consumption_df["Date"]

    consumption_df.to_csv("data/interim/consumption_df_interim.csv", index=False)
