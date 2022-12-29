"""Configuration for the pytest test suite."""

import pytest
import pandas as pd
from typing import Callable, Generator
from itertools import product


@pytest.fixture
def load_temperature(data: dict[str, pd.DataFrame]) -> Generator[dict[str, pd.DataFrame], None, None]:
    def _load_temperature_gen(data: dict[str, pd.DataFrame]) -> Generator[dict[str, pd.DataFrame], None, None]:
        load = data["load"]
        temperature = data["temperature"]
        zones = load["zone_id"].unique()
        stations = temperature["station_id"].unique()
        zones_stations = product(zones, stations)
        for zone, station in zones_stations:
            print(zone, station)
            yield {
                "load": load.loc[load["zone_id"] == zone, :],
                "temperature": temperature.loc[temperature["station_id"] == station, :],
            }

    return _load_temperature_gen(data=data)
