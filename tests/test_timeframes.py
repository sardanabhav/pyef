""" Tests for timeframes. """

import pytest
import pandas as pd
from pyef.timeframes import EnergyTimeFrame
from pyef.datasets import gefcom_load_2012, bigdeal_final_2022, bigdeal_qualifying_2022
from typing import Iterator


# bigdeal_qualifying_2022(), bigdeal_final_2022(), gefcom_load_2012()
@pytest.mark.parametrize("data", [bigdeal_qualifying_2022(), bigdeal_final_2022(), gefcom_load_2012()])
def test_freq(load_temperature: Iterator[dict[str, pd.DataFrame]]) -> None:
    for a in load_temperature:
        etf = EnergyTimeFrame(a["load"], a["temperature"])
        # assert etf.validated == Trues
        assert etf.freq_kwh == 60
        assert etf.freq_weather == 60
