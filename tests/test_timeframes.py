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


@pytest.mark.slow
@pytest.mark.parametrize(
    "data", [bigdeal_qualifying_2022(), bigdeal_final_2022(), gefcom_load_2012()]
)
def test_freq_complete(
    load_temperature_complete: Iterator[dict[str, pd.DataFrame]]
) -> None:
    for a in load_temperature_complete:
        etf = EnergyTimeFrame(a["load"], a["temperature"])
        # assert etf.validated == Trues
        assert etf.freq_kwh == 60
        assert etf.freq_weather == 60
