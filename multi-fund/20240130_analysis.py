import qlib
from qlib.constant import REG_CN

from qlib.utils import init_instance_by_config

from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord,SigAnaRecord
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 配置数据
train_period = ("2019-01-01", "2021-12-31")
valid_period = ("2022-01-01", "2022-12-31")
test_period = ("2023-01-01", "2023-08-24")

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
bench: pd.DataFrame = D.features(["SZ160706"], fields=["$close/Ref($close,1)-1"],start_time=test_period[0], end_time=test_period[1])
bench: pd.Series = bench.droplevel(level=0).iloc[:, 0]

####################################################################
#选取BETA60因子作为例子进行测试
col_name = 'BETA60'
BETA60 = test1.loc[:, col_name]

feature_df: pd.DataFrame = pd.concat((next_ret, BETA60), axis=1)
feature_df.columns = pd.MultiIndex.from_tuples(
    [("label", "next_ret"), ("feature", "BETA60")]
)

feature_df.head()

score_df:pd.DataFrame = feature_df.dropna().copy()
score_df.columns = ['label','score']

#-*- coding : utf-8-*-
import sys
sys.path.append("C://Users//huangtuo//QuantsPlaybook-master//B-因子构建类//凸显理论STR因子//")
from scr.core import calc_sigma, calc_weight
from scr.factor_analyze import clean_factor_data, get_factor_group_returns
from scr.qlib_workflow import run_model
from scr.plotting import model_performance_graph, report_graph

#生成因子分析图
model_performance_graph(score_df)



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


#按某日所有因子的IC值排序
dates = ['2023-01-03']
df_0301 = test2.loc[dates]

# 选取2023-01-03数据

columns = df_0301.columns

# 获取所有列名

sorted_columns = sorted(columns, key=lambda x: min(df_0301[x]),reverse=True)

# 根据列最小值排序列名

df_sorted = df_0301[sorted_columns]

# 使用DataFrame模式显示排序结果

df_sorted

#查看单个因子BETA60的排名，及next_ret的排名
BETA60_ret:pd.DataFrame = pd.concat((next_ret, BETA60), axis=1)
BETA60_rows = BETA60_ret.loc['2023-01-03'].nlargest(10, 'BETA60')
rows_ret = BETA60_ret.loc['2023-01-03'].nlargest(10, 'next_ret')

######################因子###############################
#按因子IC值*因子实际值生成一个数据
dates = ['2023-01-03']
fetch_factor_exp = fetch_factor.loc[dates]

#fetch_factor_exp = fetch_factor_exp.apply(lambda x: x * 100, axis=1)
test20240130_1=fetch_factor_exp*df_sorted
test20240130_1 = test20240130_1.abs()
totel_exp=test20240130_1.agg('sum', axis=1)

dates = ['2023-01-03']
next_ret_first=next_ret.loc[dates]
totel_ret:pd.DataFrame = pd.concat((next_ret_first, totel_exp), axis=1)
totel_ret.columns = ['next_ret', 'score']
totel_ret.nlargest(5, 'score')
totel_ret.nlargest(5, 'next_ret')


######################按规则进行因子筛选，优化选股结果##################################
def calculate_top_10_1ast_10_ret(df_sorted, fetch_factor, next_ret):
    first_10_columns = df_sorted.iloc[:, :10]
    last_10_columns = df_sorted.iloc[:, -10:]

    # Merge first_10_columns and last_10_columns
    merged_columns = pd.concat([first_10_columns, last_10_columns], axis=1)

    # Generate a data using factor IC value multiplied by factor actual value
    dates = ['2023-01-03']
    fetch_factor_exp = fetch_factor.loc[dates]
    fetch_factor_exp[merged_columns.columns]

    top_10_1ast_10 = fetch_factor_exp[merged_columns.columns] * merged_columns
    top_10_1ast_10 = top_10_1ast_10.abs()
    top_10_1ast_10_exp = top_10_1ast_10.agg('sum', axis=1)

    dates = ['2023-01-03']
    next_ret_first = next_ret.loc[dates]
    top_10_1ast_10_ret = pd.concat((next_ret_first, top_10_1ast_10_exp), axis=1)
    top_10_1ast_10_ret.columns = ['next_ret', 'score']

    return top_10_1ast_10_ret


top_10_1ast_10_ret = calculate_top_10_1ast_10_ret(df_sorted, fetch_factor, next_ret)
top_10_1ast_10_ret.nlargest(5, 'score')
top_10_1ast_10_ret.nlargest(5, 'next_ret')


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
        [benchmark_old],
        fields=["$close"],
        start_time=start_time,
        end_time=end_time,
    ).reset_index(level=0, drop=True)

    return data, benchmark


data,benchmark = get_backtest_data(top_10_1ast_10_ret[['score']],test_period[0],test_period[1],market)
benchmark_ret:pd.Series = benchmark['$close'].pct_change()


# 排名按日期分组生成rank列
def rank_by_date(data):
    # 复制一个工作 DataFrame
    df = data.copy()

    # 分组并对score列排名
    df = df.sort_values(['datetime', 'score'], ascending=[True, False])
    df['rank'] = df.groupby('datetime')['score'].rank(method='dense')

    return df

##################调用改写的backtrader回测函数##########################
#导入hugos_toolkit库需要指定目录
import sys
sys.path.append('C://Local_library')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
bt_result = get_backtesting(
    data,
    strategy=StockSelectStrategy,
    mulit_add_data=True,
    feedsfunc=AddSignalData,
    strategy_params={"selnum": 5, "pre": 0.05,'ascending':False,'show_log':False},
)

##############重新SignalStrategy函数##########
###self.signal = self.data.score  score数字改为排名
#修改next函数 直接判断self.signal[0]是否在排名前10的待选股票里

#self.signal[0] <= self.params.close_threshold
#and self.signal[-1] <= self.params.close_threshold