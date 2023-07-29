"""For Recency like feature exploration in an easy way.
We would include running stuff in parallel and also
support exploring the search space in different ways.

Recency Algorithms example:
1 - Brute force: run features in a loop. would be automatically
support parallel execution
2 - Hueristic: Run for n lags for fixed MA (0) to get the best lag.
Using the lag obtained, run for m MA values
3 - GA: search space exploration using genetic algorithms
4 - PSO: search space exploration using particle swarm optimization

TODO: Think about other feature explorations that can be done
"""

from datetime import datetime

from pyef import EnergyTimeFrame, ModelGridSearch
from pyef._config import get_option
from pyef.forecaster import Regressor


class Recency:
    """This implements muultiple algorithms to explore optimal lags and
    moving averages.
    """

    def __init__(
        self,
        data: EnergyTimeFrame,
        model: Regressor,
        pred_start_dates: list[datetime],
        horizon: int,
        max_lags: int = 24,
        max_ma: int = 24 * 7,
        # min_lags: int = 0,
        # min_ma: int = 0,
    ) -> None:
        """_summary_.

        Args:
            data (EnergyTimeFrame): ETF object
            model (Regressor): sklearn compatible model
            pred_start_dates (list[datetime]): list of start dates for prediction
            horizon (int): horizon for prediction
            max_lags (int, optional): max no. lags to explore. check frequency of the data before setting this. Defaults to 24.
            max_ma (int, optional): max no. of moving average. check frequency of the data before setting this. Defaults to 24 * 7.
        """
        self.data = data
        self.model = model
        self.max_lags = max_lags
        self.max_ma = max_ma
        self.pred_start_dates = pred_start_dates
        self.horizon = horizon

    def get_formula_vanilla_recency(self, lag: int, ma: int) -> str:
        vanilla = """net_load ~ trend + C(month) + C(hour) + C(day_of_week) * C(hour)
            + (Temp + Temp_2 + Temp_3) * C(hour)
            + (Temp + Temp_2 + Temp_3) * C(month)
          """
        if lag != 0 and ma != 0:
            formula = f"""{vanilla}
                    + (Temp_lag_{lag} + Temp_2_lag_{lag} + Temp_3_lag_{lag}) * C(hour)
                    + (Temp_lag_{lag} + Temp_2_lag_{lag} + Temp_3_lag_{lag}) * C(month)
                    + (Temp_ma_{ma} + Temp_2_ma_{ma} + Temp_3_ma_{ma}) * C(hour)
                    + (Temp_ma_{ma} + Temp_2_ma_{ma} + Temp_3_ma_{ma}) * C(month)
                    """
        elif lag == 0 and ma != 0:
            formula = f"""{vanilla}
                    + (Temp_ma_{ma} + Temp_2_ma_{ma} + Temp_3_ma_{ma}) * C(hour)
                    + (Temp_ma_{ma} + Temp_2_ma_{ma} + Temp_3_ma_{ma}) * C(month)
                    """
        elif lag != 0 and ma == 0:
            formula = f"""{vanilla}
                    + (Temp_lag_{lag} + Temp_2_lag_{lag} + Temp_3_lag_{lag}) * C(hour)
                    + (Temp_lag_{lag} + Temp_2_lag_{lag} + Temp_3_lag_{lag}) * C(month)
                    """
        elif lag == 0 and ma == 0:
            formula = vanilla
        else:
            return vanilla
        return formula

    def naive_method(self) -> None:
        formulae_vanilla_recency = [
            self.get_formula_vanilla_recency(lag, ma) for lag in range(0, 25) for ma in range(0, 8)
        ]
        self.grid_search = ModelGridSearch(
            data=self.data,
            models=[self.model],
            formulae=formulae_vanilla_recency,
            pred_start_dates=self.pred_start_dates,
            horizon=self.horizon,
            result_dir=get_option("log_dir"),
            n_jobs=-1,
        )
