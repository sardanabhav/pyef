"""
Forecaster class.
TODO add shit.

"""

from datetime import datetime, timedelta
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from patsy import dmatrices
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pyef.logger import get_logger
from pyef.timeframes import EnergyTimeFrame

Regressor = Union[LinearRegression, GradientBoostingRegressor, RandomForestRegressor]


logger = get_logger(__name__)


class Forecaster:
    def __init__(
        self,
        data: EnergyTimeFrame,
        formula: str,
        model: Regressor,
        pred_start: datetime,
        fit_values: bool = False,
        horizon: int = 24,
        wieghed: bool = False,
        log_target: bool = False,
        pca: bool = False,
        pca_components: int | None = None,
    ) -> None:
        self.data = data
        if self.data.validated:
            self.feature_dataset = self.data.feature_dataset.copy(deep=True)
            self.formula = formula
            self.model = model
            self.pred_start = pred_start
            self.fit_values = fit_values
            self.horizon = horizon
            self.wieghed = wieghed
            self.log_target = log_target
            self.pca = pca
            self.pca_components = pca_components
            self.pred_end = self.pred_start + timedelta(hours=self.horizon)
            self._data_prep()
        self.trained = False
        self.predicted = False
        logger.debug("forecaster initiated")

    def get_forecast(self) -> None:
        if not self.data.validated:
            raise

        if self.fit_values:
            self.X_train.columns = list(range(self.X_train.shape[1]))
            self.X_test.columns = list(range(self.X_test.shape[1]))

        if self.log_target:
            try:
                target = np.log(self.y_train)
            except:
                logger.warning("couldn't log transform target")
                self.log_target = False
                target = self.y_train
        else:
            target = self.y_train
        # TODO remove .values from predict and train
        logger.debug("fitting model")
        logger.debug(f"different cols = {list(set(list(self.X_train.columns)) - set(list(self.X_test.columns)))}")
        try:
            # logger.debug(f"train features = {list(self.X_train.columns)}")
            if not self.wieghed:
                self.model.fit(X=self.X_train, y=target)
            else:
                self.model.fit(X=self.X_train, y=target, sample_weight=self.wieghts)
            self.trained = True
        except Exception as e:
            logger.error(f"Could not train the model \n{e}")
            raise

        if self.trained:
            # logger.debug(f"test features = {list(self.X_test.columns)}")
            try:
                self.pred = pd.DataFrame(self.model.predict(X=self.X_test), index=self.X_test.index)
                self.pred.columns = ["forecast"]
                self.pred["actuals"] = self.y_test
                if self.log_target:
                    self.pred["forecast"] = np.exp(self.pred["forecast"])
            except Exception as e:
                logger.error(e)
                logger.debug(f"test features = {list(self.X_test.columns)}")
                logger.debug(f"train features = {list(self.X_train.columns)}")
                raise
            try:
                self.pred_in_sample = pd.DataFrame(self.model.predict(X=self.X_train), index=self.X_train.index)
                self.pred_in_sample.columns = ["forecast"]
                self.pred_in_sample["actuals"] = self.y_train
                if self.log_target:
                    self.pred_in_sample["forecast"] = np.exp(self.pred_in_sample["forecast"])
            except Exception as e:
                logger.error(e)
                logger.debug(f"X_train features = {list(self.X_train.columns)}")
                raise
            self.predicted = True
        else:
            logger.error(f"model not trained. Please call <obj>.train first")
            raise NotImplementedError

    def get_forecast_grouped(self, group_on: str | None = None) -> None:
        self.pred = pd.DataFrame()
        self.pred_in_sample = pd.DataFrame()

        if self.fit_values:
            self.X_train.columns = list(range(self.X_train.shape[1]))
            self.X_test.columns = list(range(self.X_test.shape[1]))

        if self.log_target:
            try:
                target = np.log(self.y_train)
            except:
                logger.warning("couldn't log transform target")
                self.log_target = False
                target = self.y_train
        else:
            target = self.y_train
        target_group = target.groupby(target.index.hour)
        X_test_group = self.X_test.groupby(self.X_test.index.hour)
        y_test_actuals = self.y_test.groupby(self.y_test.index.hour)
        if self.wieghed:
            wieght_group = self.wieghts.groupby(self.wieghts.index.hour)
        for name, group in self.X_train.groupby(self.X_train.index.hour):
            try:
                # logger.debug(f"train features = {list(self.X_train.columns)}")
                if not self.wieghed:
                    self.model.fit(X=group, y=target_group.get_group(name))
                else:
                    self.model.fit(X=group, y=target_group.get_group(name), sample_weight=wieght_group.get_group(name))
                self.trained = True
            except Exception as e:
                logger.error(f"Could not train the model \n{e}")
                raise

            if self.trained:
                # logger.debug(f"test features = {list(self.X_test.columns)}")
                test_group = X_test_group.get_group(name)
                test_actuals = y_test_actuals.get_group(name)
                try:
                    pred_group = pd.DataFrame(self.model.predict(X=test_group), index=test_group.index)
                    pred_group.columns = ["forecast"]
                    pred_group["actuals"] = test_actuals.values
                    if self.log_target:
                        pred_group["forecast"] = np.exp(pred_group["forecast"])
                    self.pred = pd.concat([self.pred, pred_group])
                except Exception as e:
                    logger.error(e)
                    logger.debug(f"test features = {list(self.X_test.columns)}")
                    logger.debug(f"train features = {list(self.X_train.columns)}")
                    raise
                try:
                    pred_group_in_sample = pd.DataFrame(self.model.predict(X=group), index=group.index)
                    pred_group_in_sample.columns = ["forecast"]
                    # pred_group_in_sample['actuals'] = self.y_train.groupby().values
                    if self.log_target:
                        pred_group_in_sample["forecast"] = np.exp(pred_group_in_sample["forecast"])
                    self.pred_in_sample = pd.concat([self.pred_in_sample, pred_group_in_sample])
                    # self.pred_in_sample = pd.DataFrame(self.model.predict(
                    #     X=group), index=group.index)
                    # self.pred_in_sample.columns = ['forecast']
                    # self.pred_in_sample['actuals'] = self.y_train
                except Exception as e:
                    logger.error(e)
                    logger.debug(f"X_train features = {list(self.X_train.columns)}")
                    raise
                self.predicted = True
            else:
                logger.error(f"model not trained. Please call <obj>.train first")
                raise NotImplementedError
        self.pred = self.pred.sort_index()
        self.pred_in_sample = self.pred_in_sample.sort_index()
        self.pred_in_sample["actuals"] = self.y_train

    def _data_prep(self) -> None:
        logger.debug("_data_prep")
        # check usage data after pred start and replace na by small value for patsy
        future_usage = self.feature_dataset.loc[self.feature_dataset.index >= self.pred_start, self.data.target_col]
        if future_usage.isna().all():
            self.feature_dataset.loc[self.feature_dataset.index > self.pred_start, self.data.target_col] = 0.001
        else:
            self.feature_dataset.loc[self.feature_dataset[self.data.target_col].isna(), self.data.target_col] = 0.01

        if self.wieghed:
            train_features, _ = self._split_data(self.feature_dataset)
            self.wieghts = train_features["importance"]

        y, X = dmatrices(formula_like=self.formula, data=self.feature_dataset, return_type="dataframe")

        if self.pca and self.pca_components:
            logger.warn(f"running PCA. Features will be reduced to {self.pca_components}")
            pca_obj = PCA(n_components=self.pca_components)
            X = pd.DataFrame(pca_obj.fit_transform(X), index=X.index)
        # else:
        #     logger.warn("Not running PCA. Something missing")
        # elif self.pca and self.pca_components == None:
        #     self.list_X = []
        #     for n in range(10, 150, 10):
        #         pca_obj = PCA(n_components=n)
        #         self.list_X.append(pca_obj.fit_transform(self.X))
        self.y_train, self.y_test = self._split_data(y)
        self.X_train, self.X_test = self._split_data(X)
        # print(self.X_train.index.min())
        # print(self.X_train.index.max())

        # print(self.X_test.index.min())
        # print(self.X_test.index.max())

    def _get_train_test(self, y: pd.DataFrame, X: pd.DataFrame) -> None:
        pass

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_split = df.copy(deep=True)
        # TODO make relativedelta configurable
        df_train = df_split.loc[
            (df_split.index >= self.pred_start - relativedelta(years=3)) & (df_split.index < self.pred_start), :
        ]
        df_test = df_split.loc[(df_split.index >= self.pred_start) & (df_split.index < self.pred_end), :]
        return df_train, df_test

    # def _split_data(
    #     self,
    #     df: pd.DataFrame
    # ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     df_split = df.copy(deep=True)
    #     df_train = df_split.loc[df_split.index < self.pred_start, :]
    #     df_test = df_split.loc[(df_split.index >= self.pred_start) &
    #                      (df_split.index < self.pred_end), :]
    #     return df_train, df_test

    def plot(self) -> Tuple[Any, Any]:
        if self.predicted:
            fig_out_of_sample = self.pred.plot()
            fig_in_sample = self.pred_in_sample.plot()
            return fig_out_of_sample, fig_in_sample
        else:
            logger.warn("Prediction not made. call get_forecast")
            return None, None
