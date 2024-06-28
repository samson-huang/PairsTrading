import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from datetime import datetime
#ranked_data.to_csv("c:\\temp\\ranked_data_20240221.csv")
ranked_data = pd.read_csv('c:\\temp\\ranked_data_20240618.csv', parse_dates=['datetime'], index_col='datetime')
#ranked_data['codename'] = ranked_data['code'].copy()
#导入hugos_toolkit库需要指定目录
import sys
import os
local_path = os.getcwd()
#local_path = "C:/Users/huangtuo/Documents\\GitHub\\PairsTrading\\multi-fund\\"
local_path = "C:/Users/huangtuo/Documents\\GitHub\\PairsTrading\\multi-fund\\"
sys.path.append(local_path+'\\Local_library\\')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
from hugos_toolkit.BackTestTemplate import LowRankStrategy_new
from typing import List, Tuple


ranked_data.loc[ranked_data['close'] <= 0, 'rank'] = 0

test_period = ("2019-01-01", "2024-05-15")
bt_result = get_backtesting(
    ranked_data,
    name="code",
    strategy=LowRankStrategy_new,
    mulit_add_data=True,
    feedsfunc=AddSignalData,
    strategy_params={"selnum": 5, "pre": 0.05, 'ascending': False, 'show_log': False},
    begin_dt=test_period[0],
    end_dt=test_period[1],
)

trade_logger = bt_result.result[0].analyzers._trade_logger.get_analysis()
TradeListAnalyzer = bt_result.result[0].analyzers._TradeListAnalyzer.get_analysis()

OrderAnalyzer = bt_result.result[0].analyzers._OrderAnalyzer.get_analysis()

trader_df = pd.DataFrame(trade_logger)
orders_df = pd.DataFrame(OrderAnalyzer)

benchmark_old = ["SH000300"]
# data, benchmark = get_backtest_data(ranked_data, test_period[0], test_period[1], market, benchmark_old)
benchmark: pd.DataFrame = D.features(
    benchmark_old,
    fields=["$close"],
    start_time=test_period[0],
    end_time=test_period[1],
).reset_index(level=0, drop=True)
benchmark_ret: pd.Series = benchmark['$close'].pct_change()

algorithm_returns: pd.Series = pd.Series(
    bt_result.result[0].analyzers._TimeReturn.get_analysis()
)
report = analysis_rets(algorithm_returns, bt_result.result, benchmark['$close'].pct_change(), use_widgets=True)

from plotly.offline import iplot
from plotly.offline import init_notebook_mode

init_notebook_mode()
for chart in report:
    iplot(chart)