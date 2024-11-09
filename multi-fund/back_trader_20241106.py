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
test_period = ("2024-07-01", "2024-11-04")

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

ranked_data = pd.read_csv('c:\\temp\\test_20241105.csv',
                          parse_dates=['datetime'],
                          index_col=['datetime', 'code'])


# 筛选出 datetime 大于 '2023-05-23' 的所有数据行
raw_data = ranked_data.loc[(ranked_data.index.get_level_values('datetime') >= '2024-07-01'), :]

import pandas as pd
#raw_data=ranked_data
# 获取所有唯一的日期和instrument
all_dates = raw_data.index.get_level_values('datetime').unique()
all_instruments = raw_data.index.get_level_values('code').unique()

# 创建一个新的MultiIndex,包含所有日期和instrument的组合
new_index = pd.MultiIndex.from_product([all_dates, all_instruments], names=['datetime', 'code'])

# 使用reindex方法重新索引DataFrame,并用0填充缺失值
new_df = raw_data.reindex(new_index, fill_value=0)

new_df = new_df.reset_index(level='code')

#剔除不在白名单里的所有股票
whitelist = pd.read_csv('C:\\temp\\important\\whitelist.csv')
new_df = new_df[new_df['code'].isin(whitelist['code'])]
##########################
# 筛选出 datetime 大于 '2023-05-23' 的所有数据行
#new_df.loc[(new_df.index.get_level_values('datetime') == '2023-05-23'), :].head(2)

bt_result = get_backtesting(
    new_df,
    name="code",
    strategy=LowRankStrategy_new,
    mulit_add_data=True,
    feedsfunc=AddSignalData,
    strategy_params={"selnum": 5, "pre": 0.05, 'ascending': False, 'show_log': False},
    begin_dt=test_period[0],
    end_dt=test_period[1],
)

benchmark_old = ["SH000300"]
# data, benchmark = get_backtest_data(ranked_data, test_period[0], test_period[1], market, benchmark_old)
benchmark: pd.DataFrame = D.features(
    benchmark_old,
    fields=["$close"],
    start_time=test_period[0],
    end_time=test_period[1],
).reset_index(level=0, drop=True)
benchmark_ret: pd.Series = benchmark['$close'].pct_change()


trade_logger = bt_result.result[0].analyzers._trade_logger.get_analysis()
TradeListAnalyzer = bt_result.result[0].analyzers._TradeListAnalyzer.get_analysis()
SellDataLogger = bt_result.result[0].analyzers._SellDataLogger.get_analysis()

OrderAnalyzer = bt_result.result[0].analyzers._OrderAnalyzer.get_analysis()

trader_df = pd.DataFrame(trade_logger)
orders_df = pd.DataFrame(OrderAnalyzer)
SellDataLogger_df  = pd.DataFrame(SellDataLogger)

algorithm_returns: pd.Series = pd.Series(
    bt_result.result[0].analyzers._TimeReturn.get_analysis()
)

#benchmark_new = benchmark[split:]
report = analysis_rets(algorithm_returns, bt_result.result, benchmark['$close'].pct_change(), use_widgets=True)

from plotly.offline import iplot
from plotly.offline import init_notebook_mode

init_notebook_mode()
for chart in report:
    iplot(chart)
#######################################################
#######################################################

i = 1
for element in TradeListAnalyzer:
    # 将整数i转换为字符串，并与路径字符串连接
    filename = f'c:\\temp\\mixed_output_{i}.csv'

    if isinstance(element, pd.DataFrame):
        # 如果元素是DataFrame，直接导出到CSV文件
        element.to_csv(filename, index=False)
        print(f"元素是DataFrame，已导出到: {filename}")
    elif isinstance(element, dict):
        # 如果元素是字典，转换为DataFrame后再导出到CSV文件
        df = pd.DataFrame([element])
        df.to_csv(filename, index=False)
        print(f"元素是字典，已转换为DataFrame并导出到: {filename}")
    else:
        print(f"元素类型为: {type(element)}, 不支持导出到CSV")

    print("/n  ---------------------------/n")

    i += 1  # 确保每次循环i的值都会增加


test1=pd.read_csv("c:\\temp\\mixed_output_1.csv")
test1 = test1.rename(columns={ '股票': 'code'})
test1['code'] = test1['code'].str.lower()
codefundsecname = pd.read_csv('c:\\temp\\important\\codefundsecname.csv')
merged_df = pd.merge( codefundsecname,test1, on='code', how='outer')
merged_df.to_csv("c:\\temp\\merged_df_20241107.csv")


trader_df = pd.DataFrame(trade_logger)
# 使用字典修改列名
codefundsecname = pd.read_csv('c:\\temp\\important\\codefundsecname.csv')
trader_df_new = trader_df.rename(columns={'buy_date': 'datetime', 'buy_name': 'code'})
trader_df_new['code'] = trader_df_new['code'].str.lower()
merged_df = pd.merge(trader_df_new, codefundsecname,on='code', how='outer')

merged_df.to_csv('c:\\temp\\trader_df_20241108.csv')
################################
###########################

'''
#############################################
#############################################
trader_df = pd.DataFrame(trade_logger)
# 使用字典修改列名
codefundsecname = pd.read_csv('c:\\temp\\upload\\codefundsecname.csv')
trader_df_new = trader_df.rename(columns={'buy_date': 'datetime', 'buy_name': 'code'})
trader_df_new['code'] = trader_df_new['code'].str.lower()
merged_df = pd.merge(trader_df_new, codefundsecname,on='code', how='outer')

merged_df.to_csv('c:\\temp\\merged_df_20241106.csv')


orders_df = pd.DataFrame(OrderAnalyzer)
orders_df=orders_df[orders_df['order_status']== 'Completed']
orders_df = orders_df.rename(columns={'order_date': 'datetime', 'order_code': 'code'})


orders_df.to_csv('c:\\temp\\orders_df_20241106.csv')
orders_df = pd.read_csv('c:\\temp\\orders_df_20241106.csv')
columns_to_drop  = ['Unnamed: 0', 'reason']
orders_df = orders_df.drop(columns_to_drop , axis=1)
#orders_df.head()

import pandas as pd

i = 1
for element in TradeListAnalyzer:
    # 将整数i转换为字符串，并与路径字符串连接
    filename = f'c:\\temp\\mixed_output_{i}.csv'

    if isinstance(element, pd.DataFrame):
        # 如果元素是DataFrame，直接导出到CSV文件
        element.to_csv(filename, index=False)
        print(f"元素是DataFrame，已导出到: {filename}")
    elif isinstance(element, dict):
        # 如果元素是字典，转换为DataFrame后再导出到CSV文件
        df = pd.DataFrame([element])
        df.to_csv(filename, index=False)
        print(f"元素是字典，已转换为DataFrame并导出到: {filename}")
    else:
        print(f"元素类型为: {type(element)}, 不支持导出到CSV")

    print("/n  ---------------------------/n")

    i += 1  # 确保每次循环i的值都会增加



test1=pd.read_csv("c:\\temp\\mixed_output_1.csv")
test1 = test1.rename(columns={ '股票': 'code'})
test1['code'] = test1['code'].str.lower()
codefundsecname = pd.read_csv('c:\\temp\\upload\\codefundsecname.csv')
merged_df = pd.merge(test1, codefundsecname,on='code', how='outer')

'''





