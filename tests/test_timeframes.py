""" Tests for timeframes. """

from typing import Iterator

import pandas as pd
import pytest

from pyef.datasets import bigdeal_final_2022, bigdeal_qualifying_2022, gefcom_load_2012
from pyef.timeframes import EnergyTimeFrame


@pytest.mark.parametrize(
    "data", [bigdeal_qualifying_2022(), bigdeal_final_2022(), gefcom_load_2012()]
)
def test_freq(load_temperature_sample: Iterator[dict[str, pd.DataFrame]]) -> None:
    for a in load_temperature_sample:
        etf = EnergyTimeFrame(a["load"], a["temperature"])
        # assert etf.validated == Trues
        assert etf.freq_kwh == 60
        assert etf.freq_weather == 60


@pytest.mark.parametrize(
    "data", [bigdeal_qualifying_2022(), bigdeal_final_2022(), gefcom_load_2012()]
)
def test_infered_series(
    load_temperature_sample: Iterator[dict[str, pd.DataFrame]]
) -> None:
    for load_temperature in load_temperature_sample:
        etf = EnergyTimeFrame(load_temperature["load"], load_temperature["temperature"])
        # assert etf.validated == Trues
        assert etf.temperature_col == "temperature"
        assert etf.target_col == "load"

        new_temperature = load_temperature["temperature"]
        new_temperature = new_temperature.rename(columns={"temperature": "t"})

        new_load = load_temperature["load"]
        new_load = new_load.rename(columns={"load": "load_kwh"})

        etf_update = EnergyTimeFrame(new_load, new_temperature)
        assert etf_update.temperature_col is None
        assert etf_update.target_col is None


@pytest.mark.slow
@pytest.mark.parametrize(
    "data", [bigdeal_qualifying_2022(), bigdeal_final_2022(), gefcom_load_2012()]
)
def test_freq_complete(
    load_temperature_complete: Iterator[dict[str, pd.DataFrame]]
) -> None:
    for load_temperature in load_temperature_complete:
        etf = EnergyTimeFrame(load_temperature["load"], load_temperature["temperature"])
        # assert etf.validated == Trues
        assert etf.freq_kwh == 60
        assert etf.freq_weather == 60
