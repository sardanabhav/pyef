""" Test Forecaster """

from typing import Iterator

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pyef.datasets import bigdeal_final_2022, bigdeal_qualifying_2022, gefcom_load_2012
from pyef.forecaster import Forecaster
from pyef.timeframes import EnergyTimeFrame

test_data = [bigdeal_qualifying_2022(), bigdeal_final_2022(), gefcom_load_2012()]
max_history_test_data = [
    # '36MS',
    # '24MS',
    # '48MS',
    "7D",
    "28D",
    "72H",
    "48H",
    "730D",
    "1095D",
]


@pytest.mark.parametrize("data", test_data)
def test_lr_sample(load_temperature_sample: Iterator[dict[str, pd.DataFrame]]) -> None:
    for load_temperature in load_temperature_sample:
        etf = EnergyTimeFrame(load_temperature["load"], load_temperature["temperature"])

        forecaster = Forecaster(
            data=etf,
            formula="load ~ C(month) + C(hour) + C(day_of_week) + temperature",
            model=LinearRegression(),
            pred_start=etf.feature_dataset.tail(8760).head(1).index[0],
            horizon=8760,
        )

        forecaster.get_forecast()
        print(forecaster.pred)
        assert forecaster.pred.shape[0] == 8760


@pytest.mark.slow
@pytest.mark.parametrize("data", test_data)
def test_lr_complete(
    load_temperature_complete: Iterator[dict[str, pd.DataFrame]]
) -> None:
    for load_temperature in load_temperature_complete:
        etf = EnergyTimeFrame(load_temperature["load"], load_temperature["temperature"])

        forecaster = Forecaster(
            data=etf,
            formula="load ~ C(month) + C(hour) + C(day_of_week) + temperature",
            model=LinearRegression(),
            pred_start=etf.feature_dataset.tail(8760).head(1).index[0],
            horizon=8760,
        )

        forecaster.get_forecast()
        print(forecaster.pred)
        assert forecaster.pred.shape[0] == 8760


@pytest.mark.parametrize("max_history", max_history_test_data)
@pytest.mark.parametrize("data", test_data)
def test_max_history(
    load_temperature_sample: Iterator[dict[str, pd.DataFrame]], max_history: str
) -> None:
    for load_temperature in load_temperature_sample:
        etf = EnergyTimeFrame(load_temperature["load"], load_temperature["temperature"])
        forecaster = Forecaster(
            data=etf,
            formula="load ~ C(month) + C(hour) + C(day_of_week) + temperature",
            model=LinearRegression(),
            pred_start=etf.feature_dataset.tail(8760).head(1).index[0],
            max_history=max_history,
            horizon=8760,
        )
        train_start_date = forecaster.y_train.index.min()
        train_stop_date = forecaster.y_test.index.min()
        freq_str = pd.tseries.frequencies.to_offset(
            train_stop_date - train_start_date
        ).freqstr
        print(train_stop_date - train_start_date)
        assert pd.tseries.frequencies.to_offset(
            freq_str
        ) == pd.tseries.frequencies.to_offset(max_history)
