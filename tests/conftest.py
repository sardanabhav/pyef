"""Configuration for the pytest test suite."""

from itertools import product
from typing import Generator

import pandas as pd
import random
import pytest


@pytest.fixture
def load_temperature_complete(
    data: dict[str, pd.DataFrame]
) -> Generator[dict[str, pd.DataFrame], None, None]:
    def _load_temperature_gen(
        data: dict[str, pd.DataFrame]
    ) -> Generator[dict[str, pd.DataFrame], None, None]:
        load = data["load"]
        temperature = data["temperature"]
        zones = load["zone_id"].unique()
        stations = temperature["station_id"].unique()
        zones_stations = product(zones, stations)
        for zone, station in zones_stations:
            yield {
                "load": load.loc[load["zone_id"] == zone, :],
                "temperature": temperature.loc[temperature["station_id"] == station, :],
            }

    return _load_temperature_gen(data=data)


@pytest.fixture
def load_temperature_sample(
    data: dict[str, pd.DataFrame]
) -> Generator[dict[str, pd.DataFrame], None, None]:
    def _load_temperature_sample_gen(
        data: dict[str, pd.DataFrame]
    ) -> Generator[dict[str, pd.DataFrame], None, None]:
        load = data["load"]
        temperature = data["temperature"]
        zones = load["zone_id"].unique()
        stations = temperature["station_id"].unique()
        zones_stations = product(zones, stations)
        selected_zone_stations = random.sample(list(zones_stations), 4)
        for zone, station in selected_zone_stations:
            yield {
                "load": load.loc[load["zone_id"] == zone, :],
                "temperature": temperature.loc[temperature["station_id"] == station, :],
            }

    return _load_temperature_sample_gen(data=data)
