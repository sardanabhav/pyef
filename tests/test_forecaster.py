""" Test Forecaster """

from typing import Iterator

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pyef.datasets import bigdeal_final_2022, bigdeal_qualifying_2022, gefcom_load_2012
from pyef.forecaster import Forecaster
from pyef.timeframes import EnergyTimeFrame


@pytest.mark.parametrize("data", [bigdeal_qualifying_2022(), bigdeal_final_2022(), gefcom_load_2012()])
def test_lr(load_temperature: Iterator[dict[str, pd.DataFrame]]) -> None:
    for a in load_temperature:
        etf = EnergyTimeFrame(a["load"], a["temperature"])

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
        # assert etf.validated == Trues
        # assert etf.freq_kwh == 60
        # assert etf.freq_weather == 60
