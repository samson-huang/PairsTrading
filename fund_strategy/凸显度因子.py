import sys
sys.path.append('C://Users//huangtuo//Documents//GitHub//PairsTrading//fund_strategy//')

import empyrical as ep
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.workflow import R  # 实验记录管理器
# from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord
from qlib.data.dataset.loader import StaticDataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.data.dataset.processor import DropnaLabel, ProcessInf, CSRankNorm, Fillna
# from qlib.utils import init_instance_by_config
from typing import List, Tuple, Dict

from scr.core import calc_sigma, calc_weight
from scr.factor_analyze import clean_factor_data, get_factor_group_returns
from scr.qlib_workflow import run_model
from scr.plotting import model_performance_graph, report_graph

import matplotlib.pyplot as plt
import seaborn as sns

# plt中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt显示负号
plt.rcParams["axes.unicode_minus"] = False

qlib.init(provider_uri="C:/Users/huangtuo/qlib_bin/" ,region="cn")

# 使用D.feature与DataLoader,DataHandlerLP,DatasetH获取数据的数据MutiIndex索引不同
# 前者Instrument,datetime后者是datetime,Instrument
POOLS: List = D.list_instruments(D.instruments("csi300"),start_time="2022-06-30", end_time="2023-06-26",as_list=True)
pct_chg: pd.DataFrame = D.features(POOLS, fields=["$close/Ref($close,1)-1"],start_time="2022-06-30", end_time="2023-06-26")
pct_chg: pd.DataFrame = pct_chg.unstack(level=0)["$close/Ref($close,1)-1"]

# 未来期收益
next_ret: pd.DataFrame = D.features(POOLS, fields=["Ref($open,-2)/Ref($open,-1)-1"],start_time="2022-06-30", end_time="2023-06-26")
next_ret.columns = ["next_ret"]
next_ret: pd.DataFrame = next_ret.swaplevel()
next_ret.sort_index(inplace=True)

# 基准
bench: pd.DataFrame = D.features(["000300.SH"], fields=["$close/Ref($close,1)-1"],start_time="2022-06-30", end_time="2023-06-26")
bench: pd.Series = bench.droplevel(level=0).iloc[:, 0]


# 计算w
w: pd.DataFrame = pct_chg.pipe(calc_sigma).pipe(calc_weight)
# 计算st因子
STR: pd.DataFrame = w.rolling(20).cov(pct_chg)

STR: pd.Series = STR.stack()
STR.name = "STR"

feature_df: pd.DataFrame = pd.concat((next_ret, STR), axis=1)
feature_df.columns = pd.MultiIndex.from_tuples(
    [("label", "next_ret"), ("feature", "STR")]
)

feature_df.head()

#因子分析
score_df:pd.DataFrame = feature_df.dropna().copy()
score_df.columns = ['label','score']

model_performance_graph(score_df)
#################################################################
#################################################################

################################################################
################################################################
# 计算获得惊恐度,准准收益使用的沪深300收益
sigma: pd.DataFrame = pct_chg.pipe(calc_sigma, bench=bench)
# 加权决策分
weighted: pd.DataFrame = sigma.mul(pct_chg)
# 加权决策分均值
avg_score: pd.DataFrame = weighted.rolling(20).mean()

avg_score_ser: pd.Series = avg_score.stack()
avg_score_ser.name = "avg_score"

# 加权决策分标准差
std_score: pd.DataFrame = weighted.rolling(20).std()

std_score_ser: pd.Series = std_score.stack()
std_score_ser.name = "std_score"

# 等权合成惊恐度得分 - 后续可以用qlib的模型合成寻找最优
terrified_score: pd.DataFrame = (avg_score + std_score) * 0.5

terrified_score_ser: pd.Series = terrified_score.stack()
terrified_score_ser.name = "terrified_score"

terrified_df: pd.DataFrame = pd.concat(
    (avg_score_ser, std_score_ser, terrified_score_ser, next_ret), axis=1
)
terrified_df.sort_index(inplace=True)

terrified_df.head()
###########################################################
#因子分析
group_returns: pd.DataFrame = (terrified_df.pipe(pd.DataFrame.dropna)
                                           .pipe(clean_factor_data)
                                           .pipe(get_factor_group_returns, quantile=5))

group_cum:pd.DataFrame = ep.cum_returns(group_returns)

# 画图
for factor_name, df in group_cum.groupby(level=0, axis=1):
    df.plot(title=factor_name, figsize=(12, 6))
    plt.axhline(0, ls="--", color="black")

#########################################################
#因子复合
test_df:pd.DataFrame = terrified_df[['avg_score','std_score','next_ret']].copy()
test_df.columns = pd.MultiIndex.from_tuples([("feature",'avg_score'),('feature','std_score'),('label',"next_ret")])



