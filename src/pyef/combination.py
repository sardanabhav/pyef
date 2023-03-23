"""
TODO design combination
Combination class for generating model combinations

Ideas:
inputs:
    * multiple forecaster objects with self.predicted = True
    validate all input models - they should have the same input etf,
    and same input forecaster parameters like pred_start and horizon

    * method - combination method to be applied, eg. simple average,
    weighted average, median, min and max pred at each timestep,
    advanced - stacking. take output of each model to train a new
    forecaster object and generate predictions
"""
