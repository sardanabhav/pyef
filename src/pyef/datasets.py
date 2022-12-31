import os

import pandas as pd
from pkg_resources import resource_filename


def gefcom_load_2012() -> dict[str, pd.DataFrame]:
    """
    Loads GEFCom 2012 dataset
    src: https://www.dropbox.com/s/epj9b57eivn79j7/GEFCom2012.zip?dl=1

    TODO add references
    TODO add examples

    """

    # TODO fix hour - hour ending (only change 24th hour to 0)
    melt_cols = ["year", "month", "day"]
    filepath = resource_filename("pyef", os.path.join("data", "gefcom2012", "load"))
    load_history = pd.read_csv(f"{filepath}/Load_history.csv", thousands=",")
    load_solution = pd.read_csv(f"{filepath}/Load_solution.csv", thousands=",").drop(["weight", "id"], axis=1)

    def convert_to_ts(df_load: pd.DataFrame) -> pd.DataFrame:
        df_load = pd.melt(df_load, id_vars=["zone_id"] + melt_cols, var_name="hour", value_name="load")
        df_load["hour"] = df_load["hour"].str.replace("h", "")
        df_load["hour"] = df_load["hour"].apply(lambda x: int(x))
        df_load["datetime"] = pd.to_datetime(df_load[["year", "month", "day", "hour"]])
        df_load = df_load.drop(["year", "month", "day", "hour"], axis=1)
        df_load = df_load.set_index("datetime").sort_index()
        return df_load

    load_history = convert_to_ts(df_load=load_history)
    load_solution = convert_to_ts(df_load=load_solution)

    df_load = (
        load_history.set_index("zone_id", append=True)
        .fillna(load_solution.set_index("zone_id", append=True))
        .reset_index(level=1)
    )

    df_temperature_1 = pd.read_csv(f"{filepath}/temperature_history.csv")

    df_temperature_1 = pd.melt(
        df_temperature_1, id_vars=["station_id"] + melt_cols, var_name="hour", value_name="temperature"
    )
    df_temperature_1["hour"] = df_temperature_1["hour"].str.replace("h", "")
    df_temperature_1["hour"] = df_temperature_1["hour"].apply(lambda x: int(x))
    df_temperature_1["datetime"] = pd.to_datetime(df_temperature_1[["year", "month", "day", "hour"]])
    df_temperature_1 = df_temperature_1.drop(["year", "month", "day", "hour"], axis=1)
    df_temperature_1 = df_temperature_1.set_index("datetime")

    df_temperature_2 = (
        pd.read_csv(f"{filepath}/temperature_solution.csv")
        .drop(["datetime", "date"], axis=1)
        .rename(columns={"T0_p1": "temperature"})
    )

    df_temperature_2["datetime"] = pd.to_datetime(df_temperature_2[["year", "month", "day", "hour"]])
    df_temperature_2 = df_temperature_2.drop(["year", "month", "day", "hour"], axis=1)
    df_temperature_2 = df_temperature_2.set_index("datetime")

    # df_temperature = pd.concat([df_temperature_1, df_temperature_2]).sort_index()
    df_temperature = (
        df_temperature_1.set_index("station_id", append=True)
        .fillna(df_temperature_2.set_index("station_id", append=True))
        .reset_index(level=1)
    ).sort_index()

    return {"load": df_load, "temperature": df_temperature}


def bigdeal_qualifying_2022() -> dict[str, pd.DataFrame]:
    """_summary_

    Returns:
        dict[str, pd.DataFrame]: _description_
    """

    filepath = resource_filename("pyef", os.path.join("data", "bigdeal2022", "qualifying_match"))

    data = pd.read_csv(f"{filepath}/data_round_1.csv")
    data.columns = data.columns.str.lower()
    data["datetime"] = pd.to_datetime(data[["year", "month", "day", "hour"]])
    data.index = data["datetime"]

    df_load = data.loc[:, ["load"]]
    df_load["zone_id"] = 1

    df_temperature_1 = data.loc[:, ["t1"]]
    df_temperature_1.columns = ["temperature"]
    df_temperature_1["station_id"] = 1

    df_temperature_2 = data.loc[:, ["t2"]]
    df_temperature_2.columns = ["temperature"]
    df_temperature_2["station_id"] = 2

    df_temperature_3 = data.loc[:, ["t3"]]
    df_temperature_3.columns = ["temperature"]
    df_temperature_3["station_id"] = 3

    df_temperature_4 = data.loc[:, ["t4"]]
    df_temperature_4.columns = ["temperature"]
    df_temperature_4["station_id"] = 4

    df_temperature = pd.concat([df_temperature_1, df_temperature_2, df_temperature_3, df_temperature_4])

    return {"load": df_load, "temperature": df_temperature}


def bigdeal_final_2022() -> dict[str, pd.DataFrame]:
    """_summary_

    Returns:
        dict[str, pd.DataFrame]: _description_
    """

    filepath = resource_filename("pyef", os.path.join("data", "bigdeal2022", "final_match"))

    data = pd.read_csv(f"{filepath}/final_match.csv")
    data.columns = data.columns.str.lower()
    data.index = pd.to_datetime(data["date"])
    data.index += pd.TimedeltaIndex(data["hour"], unit="h")
    data.index.name = "datetime"

    df_load_1 = data.loc[:, ["ldc1"]]
    df_load_1.columns = ["load"]
    df_load_1["zone_id"] = 1

    df_load_2 = data.loc[:, ["ldc2"]]
    df_load_2.columns = ["load"]
    df_load_2["zone_id"] = 2

    df_load_3 = data.loc[:, ["ldc3"]]
    df_load_3.columns = ["load"]
    df_load_3["zone_id"] = 3

    df_load = pd.concat(
        [
            df_load_1,
            df_load_2,
            df_load_3,
        ]
    )

    df_temperature_1 = data.loc[:, ["t1"]]
    df_temperature_1.columns = ["temperature"]
    df_temperature_1["station_id"] = 1

    df_temperature_2 = data.loc[:, ["t2"]]
    df_temperature_2.columns = ["temperature"]
    df_temperature_2["station_id"] = 2

    df_temperature_3 = data.loc[:, ["t3"]]
    df_temperature_3.columns = ["temperature"]
    df_temperature_3["station_id"] = 3

    df_temperature_4 = data.loc[:, ["t4"]]
    df_temperature_4.columns = ["temperature"]
    df_temperature_4["station_id"] = 4

    df_temperature_5 = data.loc[:, ["t5"]]
    df_temperature_5.columns = ["temperature"]
    df_temperature_5["station_id"] = 5

    df_temperature_6 = data.loc[:, ["t6"]]
    df_temperature_6.columns = ["temperature"]
    df_temperature_6["station_id"] = 6

    df_temperature = pd.concat(
        [
            df_temperature_1,
            df_temperature_2,
            df_temperature_3,
            df_temperature_4,
            df_temperature_5,
            df_temperature_6,
        ]
    )

    df_temperature_forecast_1 = data.loc[:, ["t1_forecast"]]
    df_temperature_forecast_1.columns = ["temperature_forecast"]
    df_temperature_forecast_1["station_id"] = 1

    df_temperature_forecast_2 = data.loc[:, ["t2_forecast"]]
    df_temperature_forecast_2.columns = ["temperature_forecast"]
    df_temperature_forecast_2["station_id"] = 2

    df_temperature_forecast_3 = data.loc[:, ["t3_forecast"]]
    df_temperature_forecast_3.columns = ["temperature_forecast"]
    df_temperature_forecast_3["station_id"] = 3

    df_temperature_forecast_4 = data.loc[:, ["t4_forecast"]]
    df_temperature_forecast_4.columns = ["temperature_forecast"]
    df_temperature_forecast_4["station_id"] = 4

    df_temperature_forecast_5 = data.loc[:, ["t5_forecast"]]
    df_temperature_forecast_5.columns = ["temperature_forecast"]
    df_temperature_forecast_5["station_id"] = 5

    df_temperature_forecast_6 = data.loc[:, ["t6_forecast"]]
    df_temperature_forecast_6.columns = ["temperature_forecast"]
    df_temperature_forecast_6["station_id"] = 6

    df_temperature_forecast = pd.concat(
        [
            df_temperature_forecast_1,
            df_temperature_forecast_2,
            df_temperature_forecast_3,
            df_temperature_forecast_4,
            df_temperature_forecast_5,
            df_temperature_forecast_6,
        ]
    )

    return {"load": df_load, "temperature": df_temperature, "temperature_forecast": df_temperature_forecast}
