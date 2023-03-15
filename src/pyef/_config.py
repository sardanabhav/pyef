import logging
import pathlib

from optioneer import Optioneer

options_maker = Optioneer()

options_maker.register_option(
    "logging_level", logging.INFO, doc="Infer type of series automatically"
)
options_maker.register_option(
    "log_dir", pathlib.Path.home().joinpath(".config", "pyef", "logs")
)
options_maker.register_option(
    "preprocessing.infer_types", True, doc="Infer type of series automatically"
)
options_maker.register_option(
    "preprocessing.infer_frequency",
    True,
    doc="Infer frequency of series automatically.\
        if false, need to set freq explicitly",
)
options_maker.register_option(
    "preprocessing.kwh.accepted_columns",
    ["usage", "solar", "wind", "kwh", "load"],
    doc="acceptable column name can contain either of these strings",
)
options_maker.register_option(
    "preprocessing.kwh.fill_na",
    "linear",
    doc="method with which nas are handeled in kwh series",
)
options_maker.register_option(
    "preprocessing.weather.accepted_columns",
    ["temperature", "wind_speed", "ghi", "humidity"],
    doc="acceptable column name can contain either of these strings",
)
options_maker.register_option(
    "preprocessing.weather.fill_na",
    "linear",
    doc="method with which nas are handeled in kwh series",
)
options_maker.register_option(
    "preprocessing.weather.hdd_ref", 58, doc="reference temperature for hdd"
)
options_maker.register_option(
    "preprocessing.weather.cdd_ref", 70, doc="reference temperature for cdd"
)
options_maker.register_option(
    "preprocessing.weather.pol_dict",
    {"temperature": [2, 3], "hdd": [2, 3], "cdd": [2, 3]},
    doc="used for _add_pol() in ETF",
)

options_maker.register_option(
    "preprocessing.weather.lags_dict",
    {
        "temperature": list(range(1, 49)),
        "temperature_2": list(range(1, 49)),
        "temperature_3": list(range(1, 49)),
    },
    doc="used for _add_lags() in ETF",
)
options_maker.register_option(
    "preprocessing.weather.mas_dict",
    {
        "temperature": list(range(1, 8)),
        "temperature_2": list(range(1, 8)),
        "temperature_3": list(range(1, 8)),
    },
    doc="used for _add_mas() in ETF",
)

options = options_maker.options
get_option = options_maker.get_option
set_option = options_maker.set_option
