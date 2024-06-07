import qlib
from qlib.constant import REG_CN
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 配置数据
#train_period = ("2019-01-01", "2021-12-31")
#valid_period = ("2022-01-01", "2022-12-31")
test_period = ("2021-01-01", "2024-05-14")

market = "filter_fund"
benchmark = "SZ160706"
from qlib.contrib.data.handler import Alpha158
dh = Alpha158(instruments='filter_fund',
              start_time=test_period[0],
              end_time=test_period[1],
              infer_processors={}
              )
#按dataframe 生成Alpha158，因子例子
test1=dh.fetch()
test1.head(2)


from qlib.data import D
from typing import List, Tuple, Dict
POOLS: List = D.list_instruments(D.instruments(market), as_list=True)

# 未来期收益
next_ret: pd.DataFrame = D.features(POOLS, fields=["Ref($close,-1)/$close-1"],start_time=test_period[0], end_time=test_period[1], freq='day')
next_ret.columns = ["next_ret"]
next_ret: pd.DataFrame = next_ret.swaplevel()
next_ret.sort_index(inplace=True)

# 基准
bench: pd.DataFrame = D.features([benchmark], fields=["$close/Ref($close,1)-1"],start_time=test_period[0], end_time=test_period[1])
bench: pd.Series = bench.droplevel(level=0).iloc[:, 0]



#######################################################################
#生成单日每个因子的IC数值
fetch_factor=dh.fetch()

def _get_score_ic_test(pred_label: pd.DataFrame):
    """

    :param pred_label:
    :return:
    """
    concat_data = pred_label.copy()
    concat_data.dropna(axis=0, how="any", inplace=True)
    _ic = concat_data.groupby(level=['datetime']).apply(
        lambda x: x["label"].corr(x["score"])
    )
    #_rank_ic = concat_data.groupby(level=['datetime']).apply(
    #    lambda x: x["label"].corr(x["score"], method="spearman")
    #)
    #return pd.DataFrame({"ic": _ic, "rank_ic": _rank_ic})
    return pd.DataFrame({"ic": _ic})

def _get_score_ic_frame(fetch_factor: pd.DataFrame):
    columns_counts = len(fetch_factor.columns)
    # 取df第一个索引层级的'datetime'值
    dates = bench.index.get_level_values(0)
    # 新建df
    score_ic_frame = pd.DataFrame(index=[])
    # 赋值给df的索引
    score_ic_frame.index = dates
    for i in range(columns_counts - 1):
        # 以KLEN参数为例子进行测算。
        fetch_factor_one = fetch_factor.iloc[:, i]
        # 取columns的名称
        columns_names = fetch_factor.columns[i]
        feature_df: pd.DataFrame = pd.concat((next_ret, fetch_factor_one), axis=1)
        feature_df.columns = pd.MultiIndex.from_tuples(
            [("label", "next_ret"), ("feature", columns_names)]
        )

        score_df: pd.DataFrame = feature_df.dropna().copy()
        score_df.columns = ['label', 'score']
        get_score_ic = _get_score_ic_test(score_df)
        # 替换单个列名
        get_score_ic = get_score_ic.rename(columns={'ic': columns_names})
        score_ic_frame = pd.concat([score_ic_frame, get_score_ic], axis=1)
    return score_ic_frame

test2: pd.DataFrame = _get_score_ic_frame(fetch_factor)
def filter_and_score(test2, fetch_factor):
    result = pd.DataFrame()
    for date_idx in fetch_factor.index.unique(level=0):
        # 所有因子单日IC值的比较筛选
        row = test2.loc[[date_idx]]  # 获取索引为date_idx的行
        fetch_factor_exp_1 = fetch_factor.loc[[date_idx]]  # 获取索引为date_idx的行

        # 过滤出列值大于0.5的列
        filtered_row = row[row.abs() > 0.5]

        # filtered_row现在是一个Series，如果你想要将其转换为DataFrame
        filtered_df = pd.DataFrame(filtered_row)

        # 删除nan值
        df_cleaned = filtered_df.dropna(axis=1, how='any')
        filtered_df_param = df_cleaned.reset_index(drop=True)
        #sorted_df = filtered_df_param.T.sort_values(by=0, ascending=False).T
        #

        fetch_factor_exp_1 = fetch_factor_exp_1.loc[:, filtered_df_param.columns]

        test20240130_1 = fetch_factor_exp_1 * row

        # 将 NaN 替换为 0
        test20240130_1 = test20240130_1.abs()
        test20240130_1 = test20240130_1.fillna(0)
        totel_exp = test20240130_1.agg('sum', axis=1)
        totel_exp = totel_exp.to_frame()
        totel_exp.columns = ['score']
        result = pd.concat([result, totel_exp])  # 将每个日期的结果合并到result中

    return result

result = filter_and_score(test2, fetch_factor)
print(result)




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



benchmark_old = ["SZ160706"]
#benchmark_old = ["SH000300"]
data,benchmark = get_backtest_data(result,test_period[0],test_period[1],market,benchmark_old)
benchmark_ret:pd.Series = benchmark['$close'].pct_change()

# 排名按日期分组生成rank列
def rank_by_date(data):
    # 复制一个工作 DataFrame
    df = data.copy()

    # 分组并对score列排名
    df = df.sort_values(['datetime', 'score'], ascending=[True, False])
    df['rank'] = df.groupby('datetime')['score'].rank(method='dense')

    return df
ranked_data = rank_by_date(data)

import sys
import os
local_path = os.getcwd()
#local_path = "C:/Users/huangtuo/Documents\\GitHub\\PairsTrading\\multi-fund\\"
sys.path.append(local_path+'\\Local_library\\')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
from hugos_toolkit.BackTestTemplate import LowRankStrategy
from typing import List, Tuple

bt_result = get_backtesting(
    ranked_data,
    name="code",
    strategy=LowRankStrategy,
    mulit_add_data=True,
    feedsfunc=AddSignalData,
    strategy_params={"selnum": 5, "pre": 0.05, 'ascending': False, 'show_log': False},
)
trade_logger = bt_result.result[0].analyzers._trade_logger.get_analysis()
TradeListAnalyzer = bt_result.result[0].analyzers._TradeListAnalyzer.get_analysis()

OrderAnalyzer = bt_result.result[0].analyzers._OrderAnalyzer.get_analysis()


#计算ATR指标
#df['TR'] = ((df['high'] - df['low']) / 2) + ((df['close'] - df['close'].shift(1)).abs() / 2)
#df['ATR'] = df['TR'].rolling(window=1).mean()





