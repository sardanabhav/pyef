"""
Contains base classes - TimeSeries and TimeFrames.

These perform preprocessing on input data and converts
it to a format which is suitable for feeding to sklearn
compatible models

"""

from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from pyef._config import get_option
from pyef.logger import get_logger

logger = get_logger(__name__)


class EnergyTimeFrame:
    """
    Class to preprocess load and temperature data to make it suitable
    for energy related time series forecasting.
    It will support electric load forecasting, electricity price forecasting,
    wind power forecasting and solar power forecasting.
    """

    def __init__(
        self,
        kwh_series: pd.Series | pd.DataFrame,
        weather_series: pd.Series | pd.DataFrame,
        ex_ante: bool = False,
        sub_hourly: bool = False,
    ) -> None:
        """TODO add description

        Args:
            kwh_series (pd.Series | pd.DataFrame): _description_
            weather_series (pd.Series | pd.DataFrame): _description_
            ex_ante (bool, optional): _description_. Defaults to False.
            sub_hourly (bool, optional): _description_. Defaults to False.
        """

        self._original_series = {
            "kwh_series": kwh_series.copy(deep=True),
            "weather_series": weather_series.copy(deep=True),
        }
        self.sub_hourly = sub_hourly
        self._validate()
        self._freq_warn = False
        if self.validated:
            self._kwh_series = kwh_series.copy(deep=True)
            self._weather_series = weather_series.copy(deep=True)

            self._infer_freq()
            self._clean_data()

            # add weather features
            # TODO Move away from adding new features. Instead, use patsy to create these features

            self._add_dd()
            self._weather_series = self._add_polynomial(
                self._weather_series, get_option("preprocessing.weather.pol_dict")
            )
            self._weather_series = self._add_lags(self._weather_series, get_option("preprocessing.weather.lags_dict"))
            self._weather_series = self._add_mas(self._weather_series, get_option("preprocessing.weather.mas_dict"))

            # create combined dataset
            self._create_feature_dataset()

            # add calendar
            self._add_calendar()
        self.cleanup()

    @property
    def original_series(self) -> dict[str, pd.DataFrame]:
        return self._original_series

    @property
    def validated(self) -> bool:
        return self._validated

    def _validate(self) -> None:
        if self._validate_series(self._original_series["kwh_series"], "kwh") & self._validate_series(
            self._original_series["weather_series"], "weather"
        ):
            self._validated = True
        else:
            self._validated = False

    def _validate_series(self, data: pd.DataFrame, series: str) -> bool:
        # Right now - this takes weather and kwh series with a valid DatetimeIndex
        # Check the column names
        cols = list(data.filter(regex="|".join(get_option(f"preprocessing.{series}.accepted_columns"))).columns)

        if data.index.inferred_type != "datetime64":
            msg = f"Could not validate {series} series. Please make sure it has a valid DatetimeIndex"
            logger.error(msg)
            return False

        # if ~ data.index.is_unique:
        #     msg = f"Could not validate {series} series. Please make sure it has unique indices"
        #     logger.error(msg)
        #     return False

        if cols == []:
            msg = f'Could not validate {series} series. Please make sure it includes one of {get_option(f"preprocessing.{series}.accepted_columns")} columns'
            logger.error(msg)
            return False

        return True

    def _io_handler(
        self,
    ) -> None:
        """
        TODO
        write  a method to handle the inputs automatically.
        start with reading csv files.
        implement reading from databases/aws s3 etc
        """

    def _get_freq(self, df: pd.DataFrame) -> int:
        freqs, counts = np.unique(
            np.array((df.index[1:] - df.index[:-1]).total_seconds() / 60).astype(int), return_counts=True
        )
        if freqs.size == 1:
            # check if infered is equal to freq infered from pandas
            pd_freq = int(pd.to_timedelta(to_offset(pd.infer_freq(df.index))).total_seconds() / 60)
            if pd_freq == freqs[0]:
                return int(freqs[0])
            else:
                # still return the infered freq, but set warn to true so that new index is created in cleaning
                logger.warning("Infered freq did not match pandas' infered freq.")
                self._freq_warn = True
                return int(freqs[0])
        elif freqs.size > 1:
            self._freq_warn = True
            logger.warning(
                "Multiple frequencies found. This might be because of some missing timestamps within the series index."
            )
            logger.warning(f"Frequencies found: {freqs}\nTheir respective counts: {counts}")
            logger.warning(f"Using freq: {freqs[np.argmax(counts)]} which has most counts {counts[np.argmax(counts)]}")
            return int(freqs[np.argmax(counts)])
        else:
            logger.error(f"{freqs}, {counts}")
            raise
        # return int((df.index[1] - df.index[0]).total_seconds() / 60)

    def _infer_freq(self) -> None:
        """
        TODO: get multiple sample frequencies and warn if not same
        """
        self._freq_kwh = self._get_freq(self._kwh_series)
        self._freq_weather = self._get_freq(self._weather_series)

    @property
    def freq_kwh(self) -> int:
        return self._freq_kwh

    @property
    def freq_weather(self) -> int:
        return self._freq_weather

    def _clean_data(self) -> None:
        # TODO Add a clean timestamp
        # self._processing_data = pd.date_range(data.raw_data.index.min(), data.raw_data.index.max(), freq='15T')
        if self._freq_warn:
            logger.warning("Recreating the datetime index for both weather and kwh series")
            # add new datetime index based on infered freq
            kwh_index = pd.date_range(
                self._kwh_series.index[0], end=self._kwh_series.index[-1], freq=f"{self.freq_kwh}min"
            )
            weather_index = pd.date_range(
                self._weather_series.index[0], end=self._weather_series.index[-1], freq=f"{self.freq_weather}min"
            )
            self._kwh_series = pd.DataFrame(index=kwh_index).merge(
                self._kwh_series, left_index=True, right_index=True, how="left"
            )
            self._kwh_series.index.name = "datetime"
            self._weather_series = pd.DataFrame(index=weather_index).merge(
                self._weather_series, left_index=True, right_index=True, how="left"
            )
            self._weather_series.index.name = "datetime"

        self._kwh_series = self._kwh_series.sort_index().interpolate(
            method=get_option(f"preprocessing.kwh.fill_na"), limit_direction="forward", axis=0
        )
        self._weather_series = self._weather_series.sort_index().interpolate(
            method=get_option(f"preprocessing.weather.fill_na"), limit_direction="forward", axis=0
        )

        if self._weather_series.shape[0] != self._kwh_series.shape[0]:
            # logger.warn("KWH and Weather series are not of the same length. Making them equal")
            if self._weather_series.index.max() > self._kwh_series.index.max():
                self._weather_series = self._weather_series.loc[
                    self._weather_series.index < self._kwh_series.index.max()
                ]
            else:
                self._kwh_series = self._kwh_series.loc[self._kwh_series.index < self._weather_series.index.max()]

    def _create_feature_dataset(self) -> None:
        self.feature_dataset = self._kwh_series.merge(
            self._weather_series, how="left", left_on=self._kwh_series.index, right_on=self._weather_series.index
        )
        self.feature_dataset = self.feature_dataset.rename(columns={"key_0": "datetime"}).fillna(method="bfill")
        self.feature_dataset.index = pd.to_datetime(self.feature_dataset["datetime"])
        # self.feature_dataset = self.feature_dataset.fillna('bfill')

    @property
    def temperature_col(self) -> str:
        # TODO update this to dynamically read the series and get temperature columns
        return "temperature"

    @property
    def target_col(self) -> str:
        # TODO update this to dynamically read the series and get temperature columns
        return "load"

    def _add_lags(
        self,
        df: pd.DataFrame,
        lags_dict: dict[str, int | list[int]],
    ) -> pd.DataFrame:
        new_cols = {}
        # TODO update frequency
        # TODO update insert logicorget rid? for creating these in forecaster
        for col, lags in lags_dict.items():
            if isinstance(lags, int):
                new_cols[f"{col}_lag_{lags}"] = df.loc[:, f"{col}"].shift(lags * int(60 / self._get_freq(df)))
            else:
                for lag in lags:
                    new_cols[f"{col}_lag_{lag}"] = df.loc[:, f"{col}"].shift(lag * int(60 / self._get_freq(df)))

        return pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    def _add_mas(
        self,
        df: pd.DataFrame,
        mas_dict: dict[str, int | list[int]],
    ) -> pd.DataFrame:
        new_cols = {}
        # TODO update frequency
        # mas in days
        mas_df = pd.DataFrame()
        for col, mas in mas_dict.items():
            if isinstance(mas, int):
                new_cols[f"{col}_ma_{mas}"] = (
                    df.loc[:, f"{col}"].rolling(24 * mas * int(60 / self._get_freq(df))).mean()
                )
            else:
                for ma in mas:
                    new_cols[f"{col}_ma_{ma}"] = (
                        df.loc[:, f"{col}"].rolling(24 * ma * int(60 / self._get_freq(df))).mean()
                    )

        return pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    def _add_polynomial(self, df: pd.DataFrame, pol_dict: dict[str, int | list[int]]) -> pd.DataFrame:
        new_cols = {}
        for col, pol in pol_dict.items():
            if isinstance(pol, int):
                new_cols[f"{col}_{pol}"] = df.loc[:, f"{col}"].pow(pol)
            else:
                for p in pol:
                    new_cols[f"{col}_{p}"] = df.loc[:, f"{col}"].pow(p)

        return pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    def _add_dd(self, method: str = "niave") -> None:
        # self._weather_series = pd.concat([self._weather_series, pd.DataFrame(self._weather_series.index, columns=['datetime'])])
        self._weather_series.insert(loc=0, column="datetime", value=self._weather_series.index)
        if method == "niave":
            daily_max_temp = pd.DataFrame(
                self._weather_series.groupby(self._weather_series.index.date)[f"{self.temperature_col}"].max()
            )
            daily_min_temp = pd.DataFrame(
                self._weather_series.groupby(self._weather_series.index.date)[f"{self.temperature_col}"].min()
            )
            daily_avg_temp = ((daily_max_temp + daily_min_temp) / 2).reset_index()
            daily_avg_temp.columns = ["datetime", "avg_temperature"]
            df_merged = self._weather_series.merge(
                daily_avg_temp["avg_temperature"],
                left_on=self._weather_series.index.date,
                right_on=daily_avg_temp["datetime"],
            )
        elif method == "daily_avg":
            daily_avg_temp = pd.DataFrame(
                self._weather_series.groupby(self._weather_series.index.date)[f"{self.temperature_col}"].mean()
            ).reset_index()
            daily_avg_temp.columns = ["datetime", "avg_temperature"]
            df_merged = self._weather_series.merge(
                daily_avg_temp["avg_temperature"],
                left_on=self._weather_series.index.date,
                right_on=daily_avg_temp["datetime"],
            )
        else:
            raise NotImplementedError

        hdd_ref_remp = get_option("preprocessing.weather.hdd_ref")
        cdd_ref_remp = get_option("preprocessing.weather.cdd_ref")

        df_merged.index = df_merged["datetime"]
        df_merged.loc[hdd_ref_remp - df_merged["avg_temperature"] >= 0, "hdd"] = (
            hdd_ref_remp - df_merged.loc[hdd_ref_remp - df_merged["avg_temperature"] > 0, "avg_temperature"]
        )
        df_merged.loc[hdd_ref_remp - df_merged["avg_temperature"] < 0, "hdd"] = 0

        df_merged.loc[df_merged["avg_temperature"] - cdd_ref_remp >= 0, "cdd"] = (
            cdd_ref_remp - df_merged.loc[df_merged["avg_temperature"] - cdd_ref_remp > 0, "avg_temperature"]
        )
        df_merged.loc[df_merged["avg_temperature"] - cdd_ref_remp < 0, "cdd"] = 0

        self._weather_series = df_merged.drop(["key_0", "datetime"], axis=1)

    def _add_calendar(self) -> None:
        self.feature_dataset["trend"] = range(self.feature_dataset.shape[0])
        self.feature_dataset["year"] = self.feature_dataset.index.year
        self.feature_dataset["month"] = self.feature_dataset.index.month
        self.feature_dataset["day_of_week"] = self.feature_dataset.index.day_of_week
        self.feature_dataset["hour"] = self.feature_dataset.index.hour
        if self.sub_hourly:
            self.feature_dataset["minute"] = self.feature_dataset.index.minute

    def cleanup(self) -> None:
        ...

    def plot(self, **kwargs: Any) -> pd.plotting.PlotAccessor:
        fig = self.feature_dataset.plot(**kwargs)
        return fig
