import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/cn_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from datetime import datetime
#ranked_data.to_csv("c:\\temp\\ranked_data_20240221.csv")
ranked_data = pd.read_csv('c:\\temp\\ranked_data_20240704.csv', parse_dates=['datetime'], index_col='datetime')
#ranked_data['codename'] = ranked_data['code'].copy()
#导入hugos_toolkit库需要指定目录
import sys
import os
local_path = os.getcwd()
#local_path = "C:/Users/huangtuo/Documents\\GitHub\\PairsTrading\\multi-fund\\"
sys.path.append(local_path+'\\Local_library\\')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
from hugos_toolkit.BackTestTemplate import LowRankStrategy_new
from typing import List, Tuple
# 配置数据
#train_period = ("2019-01-01", "2021-12-31")
#valid_period = ("2022-01-01", "2022-12-31")
test_period = ("2023-05-23", "2024-05-15")
############################backtrader数据准备#############################################
def get_backtest_data(
    pred_df: pd.DataFrame, start_time: str, end_time: str,market='market',benchmark_old='all_fund'
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # 定义股票池
    stockpool: List = D.instruments(market=market)
    # 获取test时段的行情原始数据
    raw_data: pd.DataFrame = D.features(
        stockpool,
        fields=["$open", "$high", "$low", "$close", "$volume"],
        start_time=start_time,
        end_time=end_time,
    )
    raw_data: pd.DataFrame = raw_data.swaplevel().sort_index()
    data: pd.DataFrame = pd.merge(
        raw_data, pred_df, how="inner", left_index=True, right_index=True
    ).sort_index()
    data.columns = data.columns.str.replace("$", "", regex=False)
    data: pd.DataFrame = data.reset_index(level=1).rename(
        columns={"instrument": "code"}
    )

    benchmark: pd.DataFrame = D.features(
        benchmark_old,
        fields=["$close"],
        start_time=start_time,
        end_time=end_time,
    ).reset_index(level=0, drop=True)

    return data, benchmark


#主函数
if __name__ == '__main__':
    bt_result = get_backtesting(
        ranked_data,
        name="code",
        strategy=LowRankStrategy_new,
        mulit_add_data=True,
        feedsfunc=AddSignalData,
        strategy_params={"selnum": 5, "pre": 0.05, 'ascending': False, 'show_log': False},
    )
    trade_logger = bt_result.result[0].analyzers._trade_logger.get_analysis()
    TradeListAnalyzer = bt_result.result[0].analyzers._TradeListAnalyzer.get_analysis()

    OrderAnalyzer = bt_result.result[0].analyzers._OrderAnalyzer.get_analysis()
    # 将 trades 列表转换为 DataFrame
    #trade_logger_df = pd.DataFrame(trade_logger['trades'])

    # 可选: 设置列名
    #rade_logger_df.columns = ['ref', 'buy_date', 'buy_price', 'buy_size', 'sell_date', 'sell_price', 'sell_size', 'pnl']

    # 打印 DataFrame
    #print(OrderAnalyzer)
    # 将订单信息列表转换为 DataFrame
    trader_df = pd.DataFrame(trade_logger)
    orders_df = pd.DataFrame(OrderAnalyzer)

    # 设置列名
    #orders_df.columns = ['ref', 'status', 'size', 'price', 'value', 'reason', 'date', 'data', 'type']

    # 打印 DataFrame
    #print(trader_df)
    #print(trader_df)
    benchmark_old = ["SH000300"]
    #data, benchmark = get_backtest_data(ranked_data, test_period[0], test_period[1], market, benchmark_old)
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