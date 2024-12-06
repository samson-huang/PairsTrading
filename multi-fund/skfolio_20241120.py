import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
import sys
import os
local_path = os.getcwd()
local_path = "C:/Users/huangtuo/Documents\\GitHub\\PairsTrading\\multi-fund\\"
sys.path.append(local_path+'\\Local_library\\')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
from hugos_toolkit.BackTestTemplate import LowRankStrategy_new
from typing import List, Tuple
qlib.init(provider_uri=provider_uri, region=REG_CN)
from datetime import datetime
from typing import List, Tuple, Dict
test_period = ("2014-01-01", "2024-11-20")

market = "all _fund"
# benchmark = "SZ16070"
benchmark = "SH000300"

# 获取test时段的行情原始数据
stockpool: List = D.instruments(market=market)
raw_data: pd.DataFrame = D.features(
    stockpool,
    fields=["$open", "$high", "$low", "$close", "$volume"],
    start_time=test_period[0],
    end_time=test_period[1],
)

close_data = raw_data['$close'].unstack(level='instrument')
close_data = close_data.rename_axis(index={'datetime': 'Date'})
# 将NaN值替换为0
close_data = close_data.fillna(0)

import numpy as np
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import InverseVolatility, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

X = close_data
X_train, X_test = train_test_split(X, test_size=0.80, shuffle=False)

model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(name="Max Sharpe"),
)
model.fit(X_train)
model.weights_

