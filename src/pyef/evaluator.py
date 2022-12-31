"""
Evaluator
"""

import math
import os
from datetime import datetime
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from memo import Runner, grid, memfile
from pkg_resources import resource_filename
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from pyef.forecaster import Forecaster
from pyef.logger import get_logger
from pyef.timeframes import EnergyTimeFrame

Regressor = Union[LinearRegression, GradientBoostingRegressor, RandomForestRegressor]


logger = get_logger(__name__)


class Evaluator:
    def __init__(self, forecaster: Forecaster) -> None:
        self.forecaster = forecaster
        if not self.forecaster.predicted:
            logger.error(f"model not trained. Please call forecaster <obj>.train before calling evaluator class")
            raise NotImplementedError
        self.in_sample()
        self.out_of_sample()

    def in_sample(self) -> None:
        self.in_sample_metrics = self.calculate_metrics(
            self.forecaster.pred_in_sample["actuals"], self.forecaster.pred_in_sample["forecast"]
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

    def calculate_metrics(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return {"mape": mape, "mae": mae, "rmse": rmse}


class ModelGridSearch:
    def __init__(
        self,
        data: EnergyTimeFrame,
        formulae: list[str],
        models: list[Regressor],
        pred_start_dates: list[datetime],
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
        self.file_path = resource_filename("pyef", os.path.join("data", "bigdeal2022"))

    def _search(self) -> None:
        out_of_sample = []
        in_sample = []
        plots = []
        for i, M in enumerate(self.formulae):
            logger.debug(f"model: \n{M}")
            for model in self.models:
                logger.debug(f"base forecaster: {model.__str__()}")
                for j, pred_start in enumerate(self.pred_start_dates):
                    logger.debug(f"pred start date: {pred_start}")
                    forecast = Forecaster(
                        data=self.data, formula=M, model=model, pred_start=pred_start, horizon=self.horizon
                    )
                    forecast.get_forecast()
                    if self.save_plots:
                        fig1, fig2 = forecast.plot()
                        plots.append(
                            {"out_of_sample": fig1, "in_sample": fig2, "M": i, "model": model.__str__(), "CV": j}
                        )

                    metrics = Evaluator(forecaster=forecast)
                    in_sample_results = metrics.in_sample_metrics
                    in_sample_results["M"] = i
                    in_sample_results["model"] = model.__str__()
                    in_sample_results["CV"] = j
                    out_of_sample_results = metrics.out_of_sample_metrics
                    out_of_sample_results["M"] = i
                    out_of_sample_results["model"] = model.__str__()
                    out_of_sample_results["CV"] = j
                    in_sample.append(in_sample_results)
                    out_of_sample.append(out_of_sample_results)

        self.grid_in_sample = pd.DataFrame(in_sample)
        self.grid_out_of_sample = pd.DataFrame(out_of_sample)
        self.figures = pd.DataFrame(plots)

    def _new_search(
        self,
    ) -> None:
        for model in self.models:

            @memfile(
                filepath=f'{self.file_path}/result_{model.__str__().replace("(", "").replace(")", "")}.jsonl', skip=True
            )
            def _execute_model(M: str, pred_start: datetime, horizon: int) -> dict[str, float]:
                forecast = Forecaster(data=self.data, formula=M, model=model, pred_start=pred_start, horizon=horizon)
                forecast.get_forecast()
                if self.save_plots:
                    prediction = forecast.pred.copy()
                    prediction["formula"] = M
                    prediction["model"] = f"{model.__str__()}"
                    prediction["pred_start"] = pred_start
                    prediction.to_csv(
                        f'{self.file_path}/predictions{model.__str__().replace("(", "").replace(")", "")}.csv', mode="a"
                    )
                metrics = Evaluator(forecaster=forecast)
                return {
                    "mape": metrics.out_of_sample_metrics["mape"],
                    "mape_in_sample": metrics.in_sample_metrics["mape"],
                }

            settings = grid(M=self.formulae, pred_start=self.pred_start_dates, horizon=[self.horizon])
            runner = Runner(backend="loky", n_jobs=self.n_jobs)
            runner.run(func=_execute_model, settings=settings, progbar=True)

    def run_search(self) -> None:
        self._new_search()

    def get_best_model(self) -> None:
        self.run_search()
        avg_out_of_sample = self.grid_out_of_sample.groupby(["M", "model"]).mean()
        selection_mae = avg_out_of_sample["mae"].idxmin()
        selection_mape = avg_out_of_sample["mape"].idxmin()
        selection_rmse = avg_out_of_sample["rmse"].idxmin()

        self.selected_model_mae = {
            "formula_index": f"{selection_mae[0]}",
            "formula": self.formulae[selection_mae[0]],
            "model": selection_mae[1],
            "mae": avg_out_of_sample["mae"].min(),
        }
        self.selected_model_mape = {
            "formula_index": f"{selection_mape[0]}",
            "formula": self.formulae[selection_mape[0]],
            "model": selection_mape[1],
            "mape": avg_out_of_sample["mape"].min(),
        }
        self.selected_model_rmse = {
            "formula_index": f"{selection_rmse[0]}",
            "formula": self.formulae[selection_rmse[0]],
            "model": selection_rmse[1],
            "rmse": avg_out_of_sample["rmse"].min(),
        }


# import numpy as np
# from memo import memlist, memfile, grid, time_taken, Runner

# data = []

# @memfile(filepath="results.jsonl")
# @memlist(data=data)
# @time_taken()
# def birthday_experiment(class_size, n_sim):
#     """Simulates the birthday paradox. Vectorized = Fast!"""
#     sims = np.random.randint(1, 365 + 1, (n_sim, class_size))
#     sort_sims = np.sort(sims, axis=1)
#     n_uniq = (sort_sims[:, 1:] != sort_sims[:, :-1]).sum(axis = 1) + 1
#     proba = np.mean(n_uniq != class_size)
#     return {"est_proba": proba}

# # declare all the settings to loop over
# settings = grid(class_size=range(20, 30), n_sim=[100, 10_000, 1_000_000])

# # use a runner to run over all the settings
# runner = Runner(backend="threading", n_jobs=-1)
# runner.run(func=birthday_experiment, settings=settings, progbar=True)
