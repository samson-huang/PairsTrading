from typing import Dict, List, Union

import empyrical as ep
import gradient_free_optimizers as gfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
import seaborn as sns
from qlib.data import D

from src import calc_icu_ma, get_backtest, runstrat

# plt中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt显示负号
plt.rcParams["axes.unicode_minus"] = False

qlib.init(provider_uri="C:/Users/huangtuo/qlib_bin/", region="cn")

OTO: str = "Ref($open,-1)/Ref($open,-1)-1"
CTC: str = "Ref($close,-1)/$close-1"
hs300: pd.DataFrame = D.features(["SH000300"], fields=["$close", OTO, CTC],start_time="2022-06-30", end_time="2023-06-26")
close_ser: pd.Series = hs300.droplevel(level=0)["$close"]
oto_ret: pd.Series = hs300.droplevel(level=0)[OTO]
ctc_ret: pd.Series = hs300.droplevel(level=0)[CTC]

start_dt = pd.to_datetime("2022-06-30")
end_dt = pd.to_datetime("2023-06-26")

rob_ser:pd.Series = calc_icu_ma(close_ser,5)
close_ser.loc[start_dt:end_dt].plot(figsize=(16, 6), label="close", color="black")
close_ser.rolling(5).mean().loc[start_dt:end_dt].plot(color='r', label="SMS")
rob_ser.loc[start_dt:end_dt].plot(label="Rob", color="#6cd35b")
plt.legend()

#########################################################################
#
all_df: pd.DataFrame = pd.concat(
    (
        calc_icu_ma(close_ser,i) for i in np.arange(5, 205, 5)
    ),
    axis=1,
)
all_df.columns = np.arange(5, 205, 5)

# 简单网格寻参
flag_frame: pd.DataFrame = (
    all_df.sub(close_ser, axis=0).mul(-1).apply(lambda x: np.where(x > 0, 1, 0))
)
cum_frame: pd.DataFrame = ep.cum_returns(flag_frame.mul(ctc_ret, axis=0))

cum_frame.iloc[-1].nlargest(10)

cum_frame[10].plot(label="nav",figsize=(14,4))
ep.cum_returns(close_ser.pct_change()).plot(label="bench", ls="--", color="darkgray")
plt.legend()

#回测
hs300: pd.DataFrame = D.features(
    ["SH000300"], fields="$open,$high,$low,$close,$volume".split(",")
)
#hs300.columns = hs300.columns.str.replace("$", "", regex=True)
hs300.rename(columns={'$open': 'open', '$high': 'high','$low': 'low', '$close': 'close', '$volume': 'volume'}, inplace=True)
hs300: pd.DataFrame = hs300.droplevel(level=0)

search_space: Dict = {"N": np.arange(5, 205, 5)}

iterations: int = 20


# func = partial(runstrat,dataset=hs300)
opt = gfo.EvolutionStrategyOptimizer(search_space)
opt.search(lambda x: runstrat(x, dataset=hs300,method='ann'), n_iter=iterations)

sns.heatmap(opt.search_data.set_index('N'),annot=True)

#N最优值是120
#opt.best_para: Dict = {"N": 120}

print(opt.best_para)

result = get_backtest(hs300,**opt.best_para)
ret = pd.Series(result[0].analyzers._TimeReturn.get_analysis())
ax = ep.cum_returns(ret).plot(figsize=(14, 4), label="Rob择时累计收益", color="r")
ep.cum_returns(hs300["close"].pct_change()).plot(
    ls="--", color="darkgray", label="沪深300累计收益", ax=ax
)
ax.set_ylabel('累计收益率')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.2%}".format(x)))
plt.legend()


#目标函数设置
def search_single_ma_para(close_ser: pd.Series, periods: Dict) -> float:
    N: int = np.int32(periods["N"])

    if N >= len(close_ser):
        raise ValueError("N is too large")

    signal: pd.Series = calc_icu_ma(close_ser, N)
    log_ret: pd.Series = np.log(close_ser / close_ser.shift(1))

    flag: pd.Series = np.tanh(close_ser - signal)
    return np.corrcoef(flag.iloc[N:], log_ret.iloc[N:])[0, 1]


def search_double_ma_para(close_ser: pd.Series, periods: Dict) -> float:
    N: int = np.int32(periods["N"])
    M: int = np.int32(periods["M"])

    # N 必须小于 M
    if N >= M:
        return np.nan

    if M >= len(close_ser):
        raise ValueError("N is too large")

    fast_ma: pd.Series = calc_icu_ma(close_ser, N)
    slow_ma: pd.Series = fast_ma.rolling(M).mean()
    log_ret: pd.Series = np.log(close_ser / close_ser.shift(1))

    flag: pd.Series = np.tanh(fast_ma - slow_ma)

    return np.corrcoef(flag.iloc[(M + N) :], log_ret.iloc[(M + N) :])[0, 1]

################
search_space: Dict = {"N": np.arange(5, 205, 5), "M": np.arange(5, 205, 5)}

iterations: int = 250

# func = partial(runstrat,dataset=hs300)
# HillClimbingOptimizer
# EvolutionStrategyOptimizer
opt = gfo.EvolutionStrategyOptimizer(search_space,population=20)
# opt.search(lambda x: search_double_ma_para(close_ser,periods=x), n_iter=iterations)
opt.search(lambda x: search_double_ma_para(close_ser,periods=x), n_iter=iterations)

search_data: pd.DataFrame = opt.search_data.copy()
search_data["Sharpe"] = 0
search_data["CumRet"] = 0
for idx, rows in search_data.iterrows():
    N = np.int16(rows["N"])
    M = np.int16(rows["M"])
    fast_ma: pd.Series = calc_icu_ma(close_ser, N)
    slow_ma: pd.Series = fast_ma.rolling(M).mean()

    flag: pd.Series = (fast_ma - slow_ma).apply(lambda x: np.where(x > 0, 1, 0))
    ret: pd.Series = flag * close_ser.pct_change().shift(-1)

    search_data.loc[idx, "Sharpe"] = ep.sharpe_ratio(ret)
    search_data.loc[idx, "CumRet"] = ep.cum_returns(ret).iloc[-1]


