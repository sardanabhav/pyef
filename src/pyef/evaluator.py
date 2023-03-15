"""
Evaluator
"""

import math
import pathlib
from datetime import datetime
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from memo import Runner, grid, memfile, memlist
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from pyef.forecaster import Forecaster
from pyef.logger import get_logger
from pyef._config import get_option
from pyef.timeframes import EnergyTimeFrame

Regressor = Union[LinearRegression, GradientBoostingRegressor, RandomForestRegressor]

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, forecaster: Forecaster) -> None:
        self.forecaster = forecaster
        if not self.forecaster.predicted:
            logger.error(
                "model not trained.\
                Please call forecaster <obj>.train before calling evaluator class"
            )
            raise NotImplementedError
        self.in_sample()
        self.out_of_sample()

    def in_sample(self) -> None:
        self.in_sample_metrics = self.calculate_metrics(
            self.forecaster.pred_in_sample["actuals"],
            self.forecaster.pred_in_sample["forecast"],
        )

    def out_of_sample(self) -> None:
        self.out_of_sample_metrics = self.calculate_metrics(
            self.forecaster.pred["actuals"], self.forecaster.pred["forecast"]
        )

    def peak_timing_error(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Any:
        error = np.abs(y_true - y_pred)
        error["weight"] = 0
        error.loc[error["hour"] == 1, "weight"] = 1
        error.loc[(error["hour"] >= 2), "weight"] = 2
        return np.clip(error["hour"] * error["weight"], a_max=10, a_min=None).sum()

    def calculate_metrics(
        self, y_true: pd.DataFrame, y_pred: pd.DataFrame
    ) -> Dict[str, float]:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # peak_timing_error = self.peak_timing_error(y_true, y_pred)
        return {
            "mape": mape,
            "mae": mae,
            "rmse": rmse,
            # "peak_timing_error": peak_timing_error
        }


class ModelGridSearch:
    def __init__(
        self,
        data: EnergyTimeFrame,
        formulae: list[str],
        models: list[Regressor],
        pred_start_dates: list[datetime],
        result_dir: str | None = None,
        save_plots: bool = False,
        horizon: int = 8760,
        n_jobs: int = 1,
    ) -> None:
        self.data = data
        self.formulae = formulae
        self.models = models
        self.pred_start_dates = pred_start_dates
        self.save_plots = save_plots
        self.horizon = horizon
        self.n_jobs = n_jobs
        self.result_dir = result_dir

    def _search(self) -> None:
        out_of_sample = []
        in_sample = []
        plots = []
        for M_index, M in enumerate(self.formulae):
            logger.debug(f"model: \n{M}")
            for model in self.models:
                logger.debug(f"base forecaster: {model.__str__()}")
                for CV_index, pred_start in enumerate(self.pred_start_dates):
                    logger.debug(f"pred start date: {pred_start}")
                    forecast = Forecaster(
                        data=self.data,
                        formula=M,
                        model=model,
                        pred_start=pred_start,
                        horizon=self.horizon,
                    )
                    forecast.get_forecast()
                    if self.save_plots:
                        plot_dict = forecast.plot()
                        plots.append(
                            {
                                "out_of_sample": plot_dict["out_of_sample"],
                                "in_sample": plot_dict["in_sample"],
                                "M": M_index,
                                "model": model.__str__(),
                                "CV": CV_index,
                            }
                        )

                    metrics = Evaluator(forecaster=forecast)
                    in_sample_results = metrics.in_sample_metrics
                    in_sample_results["M"] = M_index
                    in_sample_results["model"] = model.__str__()
                    in_sample_results["CV"] = CV_index
                    out_of_sample_results = metrics.out_of_sample_metrics
                    out_of_sample_results["M"] = M_index
                    out_of_sample_results["model"] = model.__str__()
                    out_of_sample_results["CV"] = CV_index
                    in_sample.append(in_sample_results)
                    out_of_sample.append(out_of_sample_results)

        self.grid_in_sample = pd.DataFrame(in_sample)
        self.grid_out_of_sample = pd.DataFrame(out_of_sample)
        self.figures = pd.DataFrame(plots)

    def _new_search(
        self,
    ) -> None:
        if self.result_dir is None:
            results: list[dict[str, Any]] = []
            memodecorator = memlist(data=results)
        else:
            self.file_path = pathlib.Path(
                get_option("log_dir").joinpath(self.result_dir)
            )
            self.file_path.mkdir(parents=True, exist_ok=True)
            results_file = self.file_path.joinpath("results.jsonl")
            # f"{self.file_path}/results.jsonl"
            memodecorator = memfile(
                filepath=results_file,
                skip=True,
            )

        @memodecorator
        def _execute_model(
            M: str, model: Regressor, pred_start: datetime, horizon: int
        ) -> dict[str, dict[str, float]]:
            model_name = model.__str__().replace("(", "").replace(")", "")
            forecast = Forecaster(
                data=self.data,
                formula=M,
                model=model,
                pred_start=pred_start,
                horizon=horizon,
            )
            forecast.get_forecast()
            if self.save_plots:
                fig_path = pathlib.Path.home().joinpath(
                    ".config",
                    "pyef",
                    "figures",
                    str(horizon),
                    str(pred_start),
                    model_name,
                    M,
                )
                pathlib.Path(fig_path).mkdir(parents=True, exist_ok=True)
                plot_dict = forecast.plot()
                plot_dict["in_sample"].write_html(
                    pathlib.Path.home().joinpath(fig_path, "in_sample.html")
                )
                plot_dict["out_of_sample"].write_html(
                    pathlib.Path.home().joinpath(fig_path, "out_of_sample.html")
                )
            metrics = Evaluator(forecaster=forecast)
            return {
                "out_of_sample": metrics.out_of_sample_metrics,
                "in_sample": metrics.in_sample_metrics,
            }

        settings = grid(
            M=self.formulae,
            model=self.models,
            pred_start=self.pred_start_dates,
            horizon=[self.horizon],
        )
        runner = Runner(backend="loky", n_jobs=self.n_jobs)
        runner.run(func=_execute_model, settings=settings, progbar=True)

        if self.result_dir is None:
            results_df = pd.DataFrame(results)
            results_df.loc[:, "model"] = (
                results_df["model"]
                .astype(str)
                .str.replace(r"(", "", regex=True)
                .str.replace(r")", "", regex=True)
            )
        else:
            results_df = pd.read_json(results_file, lines=True)
        self.grid_in_sample = results_df.loc[
            :, ["M", "pred_start", "horizon", "model"]
        ].join(pd.json_normalize(results_df.pop("in_sample")))
        self.grid_out_of_sample = results_df.loc[
            :, ["M", "pred_start", "horizon", "model"]
        ].join(pd.json_normalize(results_df.pop("out_of_sample")))
        # self.figures = pd.DataFrame(plots)

    def run_search(self) -> None:
        self._new_search()

    def get_best_model(self) -> None:
        self.run_search()
        avg_out_of_sample = (
            self.grid_out_of_sample.loc[:, ["M", "model", "mape", "mae", "rmse"]]
            .groupby(["M", "model"])
            .mean(numeric_only=True)
        )
        selection_mae = avg_out_of_sample["mae"].idxmin()
        selection_mape = avg_out_of_sample["mape"].idxmin()
        selection_rmse = avg_out_of_sample["rmse"].idxmin()

        self.selected_model_mae = {
            "formula": selection_mae[0],
            "model": selection_mae[1],
            "mae": avg_out_of_sample["mae"].min(),
        }
        self.selected_model_mape = {
            "formula": selection_mape[0],
            "model": selection_mape[1],
            "mape": avg_out_of_sample["mape"].min(),
        }
        self.selected_model_rmse = {
            "formula": selection_rmse[0],
            "model": selection_rmse[1],
            "rmse": avg_out_of_sample["rmse"].min(),
        }
