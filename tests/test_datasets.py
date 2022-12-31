""" Tests for loading datasets. """

from pyef.datasets import bigdeal_final_2022, bigdeal_qualifying_2022, gefcom_load_2012


def test_gefcom_load_2012_keys() -> None:
    gefcom_2012 = gefcom_load_2012()
    assert "load" in gefcom_2012
    assert "temperature" in gefcom_2012


def test_gefcom_load_2012_indices() -> None:
    gefcom_2012 = gefcom_load_2012()
    assert gefcom_2012["load"].set_index("zone_id", append=True).index.duplicated().sum() == 0
    assert gefcom_2012["temperature"].set_index("station_id", append=True).index.duplicated().sum() == 0


def test_gefcom_load_2012_cols() -> None:
    gefcom_2012 = gefcom_load_2012()
    load = gefcom_2012["load"]
    temperature = gefcom_2012["temperature"]
    assert load.columns.str.contains("load|zone").sum() == load.shape[1]
    assert temperature.columns.str.contains("temperature|station").sum() == temperature.shape[1]
    assert load.index.name == temperature.index.name == "datetime"


def test_gefcom_load_2012_shape() -> None:
    gefcom_2012 = gefcom_load_2012()
    load = gefcom_2012["load"]
    temperature = gefcom_2012["temperature"]
    zones = load.groupby("zone_id")
    stations = temperature.groupby("station_id")
    assert (zones.size().unique() == (39600)).all()
    assert (stations.size().unique() == (39432)).all()


def test_bigdeal_qualifying_2022_keys() -> None:
    bigdeal_2022 = bigdeal_qualifying_2022()
    assert "load" in bigdeal_2022
    assert "temperature" in bigdeal_2022


def test_bigdeal_qualifying_2022_cols() -> None:
    bigdeal_2022 = bigdeal_qualifying_2022()
    load = bigdeal_2022["load"]
    temperature = bigdeal_2022["temperature"]
    assert load.columns.str.contains("load|zone").sum() == load.shape[1]
    assert temperature.columns.str.contains("temperature|station").sum() == temperature.shape[1]
    assert load.index.name == temperature.index.name == "datetime"


def test_bigdeal_qualifying_2022_shape() -> None:
    bigdeal_2022 = bigdeal_qualifying_2022()
    load = bigdeal_2022["load"]
    temperature = bigdeal_2022["temperature"]
    stations = temperature.groupby("station_id")
    assert load.shape[0] == 52584
    assert (stations.size().unique() == (52584)).all()


def test_bigdeal_final_2022_keys() -> None:
    bigdeal_2022 = bigdeal_final_2022()
    assert "load" in bigdeal_2022
    assert "temperature" in bigdeal_2022


def test_bigdeal_final_2022_cols() -> None:
    bigdeal_2022 = bigdeal_final_2022()
    load = bigdeal_2022["load"]
    temperature = bigdeal_2022["temperature"]
    assert load.columns.str.contains("load|zone_id").sum() == load.shape[1]
    assert temperature.columns.str.contains("temperature|station").sum() == temperature.shape[1]
    assert load.index.name == temperature.index.name == "datetime"


def test_bigdeal_final_2022_shape() -> None:
    bigdeal_2022 = bigdeal_final_2022()
    load = bigdeal_2022["load"]
    temperature = bigdeal_2022["temperature"]
    zones = load.groupby("zone_id")
    stations = temperature.groupby("station_id")
    assert (zones.size().unique() == (35064)).all()
    assert (stations.size().unique() == (35064)).all()
