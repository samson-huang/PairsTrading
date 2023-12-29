import qlib
from qlib.constant import REG_CN

from qlib.utils import init_instance_by_config

from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import TSDatasetH
from qlib.contrib.model.pytorch_alstm_ts import ALSTM

provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 配置数据
train_period = ("2019-01-01", "2021-12-31")
valid_period = ("2022-01-01", "2022-12-31")
test_period = ("2023-01-01", "2023-08-24")



market = "filter_fund"
benchmark = "SZ160706"

dh = Alpha158(instruments='filter_fund',
              start_time=test_period[0],
              end_time=test_period[1],
              infer_processors={}
              )

print(dh.get_cols())
'''
['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 
 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 
 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 
 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 
 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 
 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 
 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 
 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 
 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 
 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 
 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 
 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 
 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 
 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60', 'LABEL0']
 '''
dh.get_feature_config()
'''['($close-$open)/$open','($high-$low)/$open','($close-$open)/($high-$low+1e-12)','($high-Greater($open, $close))/$open',
  '($high-Greater($open, $close))/($high-$low+1e-12)','(Less($open, $close)-$low)/$open','(Less($open, $close)-$low)/($high-$low+1e-12)',
  '(2*$close-$high-$low)/$open','(2*$close-$high-$low)/($high-$low+1e-12)',
  '$open/$close','$high/$close','$low/$close','$vwap/$close',
  'Ref($close, 5)/$close','Ref($close, 10)/$close','Ref($close, 20)/$close','Ref($close, 30)/$close','Ref($close, 60)/$close',
  'Mean($close, 5)/$close','Mean($close, 10)/$close','Mean($close, 20)/$close','Mean($close, 30)/$close','Mean($close, 60)/$close',
  'Std($close, 5)/$close','Std($close, 10)/$close','Std($close, 20)/$close','Std($close, 30)/$close','Std($close, 60)/$close',
  'Slope($close, 5)/$close','Slope($close, 10)/$close','Slope($close, 20)/$close','Slope($close, 30)/$close','Slope($close, 60)/$close',
  'Rsquare($close, 5)','Rsquare($close, 10)','Rsquare($close, 20)','Rsquare($close, 30)','Rsquare($close, 60)',
  'Resi($close, 5)/$close','Resi($close, 10)/$close','Resi($close, 20)/$close','Resi($close, 30)/$close','Resi($close, 60)/$close',
  'Max($high, 5)/$close','Max($high, 10)/$close','Max($high, 20)/$close','Max($high, 30)/$close','Max($high, 60)/$close',
  'Min($low, 5)/$close','Min($low, 10)/$close','Min($low, 20)/$close','Min($low, 30)/$close','Min($low, 60)/$close',
  'Quantile($close, 5, 0.8)/$close','Quantile($close, 10, 0.8)/$close','Quantile($close, 20, 0.8)/$close','Quantile($close, 30, 0.8)/$close','Quantile($close, 60, 0.8)/$close',
  'Quantile($close, 5, 0.2)/$close','Quantile($close, 10, 0.2)/$close','Quantile($close, 20, 0.2)/$close','Quantile($close, 30, 0.2)/$close','Quantile($close, 60, 0.2)/$close',
  'Rank($close, 5)','Rank($close, 10)','Rank($close, 20)','Rank($close, 30)','Rank($close, 60)',
  '($close-Min($low, 5))/(Max($high, 5)-Min($low, 5)+1e-12)',
  '($close-Min($low, 10))/(Max($high, 10)-Min($low, 10)+1e-12)',
  '($close-Min($low, 20))/(Max($high, 20)-Min($low, 20)+1e-12)',
  '($close-Min($low, 30))/(Max($high, 30)-Min($low, 30)+1e-12)',
  '($close-Min($low, 60))/(Max($high, 60)-Min($low, 60)+1e-12)',
  'IdxMax($high, 5)/5',
  'IdxMax($high, 10)/10',
  'IdxMax($high, 20)/20',
  'IdxMax($high, 30)/30',
  'IdxMax($high, 60)/60',
  'IdxMin($low, 5)/5',
  'IdxMin($low, 10)/10',
  'IdxMin($low, 20)/20',
  'IdxMin($low, 30)/30',
  'IdxMin($low, 60)/60',
  '(IdxMax($high, 5)-IdxMin($low, 5))/5',
  '(IdxMax($high, 10)-IdxMin($low, 10))/10',
  '(IdxMax($high, 20)-IdxMin($low, 20))/20',
  '(IdxMax($high, 30)-IdxMin($low, 30))/30',
  '(IdxMax($high, 60)-IdxMin($low, 60))/60',
  'Corr($close, Log($volume+1), 5)',
  'Corr($close, Log($volume+1), 10)',
  'Corr($close, Log($volume+1), 20)',
  'Corr($close, Log($volume+1), 30)',
  'Corr($close, Log($volume+1), 60)',
  'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)',
  'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)',
  'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 20)',
  'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 30)',
  'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)',
  'Mean($close>Ref($close, 1), 5)',
  'Mean($close>Ref($close, 1), 10)',
  'Mean($close>Ref($close, 1), 20)',
  'Mean($close>Ref($close, 1), 30)',
  'Mean($close>Ref($close, 1), 60)',
  'Mean($close<Ref($close, 1), 5)',
  'Mean($close<Ref($close, 1), 10)',
  'Mean($close<Ref($close, 1), 20)',
  'Mean($close<Ref($close, 1), 30)',
  'Mean($close<Ref($close, 1), 60)',
  'Mean($close>Ref($close, 1), 5)-Mean($close<Ref($close, 1), 5)',
  'Mean($close>Ref($close, 1), 10)-Mean($close<Ref($close, 1), 10)',
  'Mean($close>Ref($close, 1), 20)-Mean($close<Ref($close, 1), 20)',
  'Mean($close>Ref($close, 1), 30)-Mean($close<Ref($close, 1), 30)',
  'Mean($close>Ref($close, 1), 60)-Mean($close<Ref($close, 1), 60)',
  'Sum(Greater($close-Ref($close, 1), 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)',
  'Sum(Greater($close-Ref($close, 1), 0), 10)/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)',
  'Sum(Greater($close-Ref($close, 1), 0), 20)/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)',
  'Sum(Greater($close-Ref($close, 1), 0), 30)/(Sum(Abs($close-Ref($close, 1)), 30)+1e-12)',
  'Sum(Greater($close-Ref($close, 1), 0), 60)/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)',
  'Sum(Greater(Ref($close, 1)-$close, 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)',
  'Sum(Greater(Ref($close, 1)-$close, 0), 10)/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)',
  'Sum(Greater(Ref($close, 1)-$close, 0), 20)/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)',
  'Sum(Greater(Ref($close, 1)-$close, 0), 30)/(Sum(Abs($close-Ref($close, 1)), 30)+1e-12)',
  'Sum(Greater(Ref($close, 1)-$close, 0), 60)/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)',
  '(Sum(Greater($close-Ref($close, 1), 0), 5)-Sum(Greater(Ref($close, 1)-$close, 0), 5))/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)',
  '(Sum(Greater($close-Ref($close, 1), 0), 10)-Sum(Greater(Ref($close, 1)-$close, 0), 10))/(Sum(Abs($close-Ref($close, 1)), 10)+1e-12)',
  '(Sum(Greater($close-Ref($close, 1), 0), 20)-Sum(Greater(Ref($close, 1)-$close, 0), 20))/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)',
  '(Sum(Greater($close-Ref($close, 1), 0), 30)-Sum(Greater(Ref($close, 1)-$close, 0), 30))/(Sum(Abs($close-Ref($close, 1)), 30)+1e-12)',
  '(Sum(Greater($close-Ref($close, 1), 0), 60)-Sum(Greater(Ref($close, 1)-$close, 0), 60))/(Sum(Abs($close-Ref($close, 1)), 60)+1e-12)',
  'Mean($volume, 5)/($volume+1e-12)',
  'Mean($volume, 10)/($volume+1e-12)',
  'Mean($volume, 20)/($volume+1e-12)',
  'Mean($volume, 30)/($volume+1e-12)',
  'Mean($volume, 60)/($volume+1e-12)',
  'Std($volume, 5)/($volume+1e-12)',
  'Std($volume, 10)/($volume+1e-12)',
  'Std($volume, 20)/($volume+1e-12)',
  'Std($volume, 30)/($volume+1e-12)',
  'Std($volume, 60)/($volume+1e-12)',
  'Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)',
  'Std(Abs($close/Ref($close, 1)-1)*$volume, 10)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 10)+1e-12)',
  'Std(Abs($close/Ref($close, 1)-1)*$volume, 20)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 20)+1e-12)',
  'Std(Abs($close/Ref($close, 1)-1)*$volume, 30)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 30)+1e-12)',
  'Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)',
  'Sum(Greater($volume-Ref($volume, 1), 0), 5)/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)',
  'Sum(Greater($volume-Ref($volume, 1), 0), 10)/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)',
  'Sum(Greater($volume-Ref($volume, 1), 0), 20)/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)',
  'Sum(Greater($volume-Ref($volume, 1), 0), 30)/(Sum(Abs($volume-Ref($volume, 1)), 30)+1e-12)',
  'Sum(Greater($volume-Ref($volume, 1), 0), 60)/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)',
  'Sum(Greater(Ref($volume, 1)-$volume, 0), 5)/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)',
  'Sum(Greater(Ref($volume, 1)-$volume, 0), 10)/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)',
  'Sum(Greater(Ref($volume, 1)-$volume, 0), 20)/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)',
  'Sum(Greater(Ref($volume, 1)-$volume, 0), 30)/(Sum(Abs($volume-Ref($volume, 1)), 30)+1e-12)',
  'Sum(Greater(Ref($volume, 1)-$volume, 0), 60)/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)',
  '(Sum(Greater($volume-Ref($volume, 1), 0), 5)-Sum(Greater(Ref($volume, 1)-$volume, 0), 5))/(Sum(Abs($volume-Ref($volume, 1)), 5)+1e-12)',
  '(Sum(Greater($volume-Ref($volume, 1), 0), 10)-Sum(Greater(Ref($volume, 1)-$volume, 0), 10))/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)',
  '(Sum(Greater($volume-Ref($volume, 1), 0), 20)-Sum(Greater(Ref($volume, 1)-$volume, 0), 20))/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)',
  '(Sum(Greater($volume-Ref($volume, 1), 0), 30)-Sum(Greater(Ref($volume, 1)-$volume, 0), 30))/(Sum(Abs($volume-Ref($volume, 1)), 30)+1e-12)',
  '(Sum(Greater($volume-Ref($volume, 1), 0), 60)-Sum(Greater(Ref($volume, 1)-$volume, 0), 60))/(Sum(Abs($volume-Ref($volume, 1)), 60)+1e-12)'],
 ['KMID',
  'KLEN',
  'KMID2',
  'KUP',
  'KUP2',
  'KLOW',
  'KLOW2',
  'KSFT',
  'KSFT2',
  'OPEN0',
  'HIGH0',
  'LOW0',
  'VWAP0',
  'ROC5',
  'ROC10',
  'ROC20',
  'ROC30',
  'ROC60',
  'MA5',
  'MA10',
  'MA20',
  'MA30',
  'MA60',
  'STD5',
  'STD10',
  'STD20',
  'STD30',
  'STD60',
  'BETA5',
  'BETA10',
  'BETA20',
  'BETA30',
  'BETA60',
  'RSQR5',
  'RSQR10',
  'RSQR20',
  'RSQR30',
  'RSQR60',
  'RESI5',
  'RESI10',
  'RESI20',
  'RESI30',
  'RESI60',
  'MAX5',
  'MAX10',
  'MAX20',
  'MAX30',
  'MAX60',
  'MIN5',
  'MIN10',
  'MIN20',
  'MIN30',
  'MIN60',
  'QTLU5',
  'QTLU10',
  'QTLU20',
  'QTLU30',
  'QTLU60',
  'QTLD5',
  'QTLD10',
  'QTLD20',
  'QTLD30',
  'QTLD60',
  'RANK5',
  'RANK10',
  'RANK20',
  'RANK30',
  'RANK60',
  'RSV5',
  'RSV10',
  'RSV20',
  'RSV30',
  'RSV60',
  'IMAX5',
  'IMAX10',
  'IMAX20',
  'IMAX30',
  'IMAX60',
  'IMIN5',
  'IMIN10',
  'IMIN20',
  'IMIN30',
  'IMIN60',
  'IMXD5',
  'IMXD10',
  'IMXD20',
  'IMXD30',
  'IMXD60',
  'CORR5',
  'CORR10',
  'CORR20',
  'CORR30',
  'CORR60',
  'CORD5',
  'CORD10',
  'CORD20',
  'CORD30',
  'CORD60',
  'CNTP5',
  'CNTP10',
  'CNTP20',
  'CNTP30',
  'CNTP60',
  'CNTN5',
  'CNTN10',
  'CNTN20',
  'CNTN30',
  'CNTN60',
  'CNTD5',
  'CNTD10',
  'CNTD20',
  'CNTD30',
  'CNTD60',
  'SUMP5',
  'SUMP10',
  'SUMP20',
  'SUMP30',
  'SUMP60',
  'SUMN5',
  'SUMN10',
  'SUMN20',
  'SUMN30',
  'SUMN60',
  'SUMD5',
  'SUMD10',
  'SUMD20',
  'SUMD30',
  'SUMD60',
  'VMA5',
  'VMA10',
  'VMA20',
  'VMA30',
  'VMA60',
  'VSTD5',
  'VSTD10',
  'VSTD20',
  'VSTD30',
  'VSTD60',
  'WVMA5',
  'WVMA10',
  'WVMA20',
  'WVMA30',
  'WVMA60',
  'VSUMP5',
  'VSUMP10',
  'VSUMP20',
  'VSUMP30',
  'VSUMP60',
  'VSUMN5',
  'VSUMN10',
  'VSUMN20',
  'VSUMN30',
  'VSUMN60',
  'VSUMD5',
  'VSUMD10',
  'VSUMD20',
  'VSUMD30',
  'VSUMD60'])

'''
test1=dh.fetch()
#取dataframe的columns的个数
print(len(test1.columns))
#取dataframe第一列数据
test1.iloc[:, 0:1]

from qlib.data import D
from typing import List, Tuple, Dict
POOLS: List = D.list_instruments(D.instruments(market), as_list=True)




# 基准
bench: pd.DataFrame = D.features(["SZ160706"], fields=["$close/Ref($close,1)-1"],start_time=test_period[0], end_time=test_period[1])
bench: pd.Series = bench.droplevel(level=0).iloc[:, 0]

#以KLEN参数为例子进行测算。
KLEN = test1.iloc[:, 1]

feature_df: pd.DataFrame = pd.concat((next_ret, KLEN), axis=1)
feature_df.columns = pd.MultiIndex.from_tuples(
    [("label", "next_ret"), ("feature", "KLEN")]
)

score_df:pd.DataFrame = feature_df.dropna().copy()
score_df.columns = ['label','score']

#-*- coding : utf-8-*-
import sys
sys.path.append("C://Users//huangtuo//QuantsPlaybook-master//B-因子构建类//凸显理论STR因子//")
from scr.core import calc_sigma, calc_weight
from scr.factor_analyze import clean_factor_data, get_factor_group_returns
from scr.qlib_workflow import run_model
from scr.plotting import model_performance_graph, report_graph
model_performance_graph(score_df)

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
get_score_ic=_get_score_ic_test(score_df)

get_score_ic.head()



#取dataframe的columns的个数
columns_counts=len(fetch_factor.columns)

for i in range(columns_counts-1):
    # 以KLEN参数为例子进行测算。
    fetch_factor_one = fetch_factor.iloc[:, i]
    #取columns的名称
    columns_names=fetch_factor.columns[i]
    feature_df: pd.DataFrame = pd.concat((next_ret, fetch_factor_one), axis=1)
    feature_df.columns = pd.MultiIndex.from_tuples(
        [("label", "next_ret"), ("feature", columns_names)]
    )

    score_df: pd.DataFrame = feature_df.dropna().copy()
    score_df.columns = ['label', 'score']
    get_score_ic = _get_score_ic_test(score_df)
    print(i)

# 未来期收益
next_ret: pd.DataFrame = D.features(POOLS, fields=["Ref($open,-2)/Ref($open,-1)-1"],start_time=test_period[0], end_time=test_period[1], freq='day')
next_ret.columns = ["next_ret"]
next_ret: pd.DataFrame = next_ret.swaplevel()
next_ret.sort_index(inplace=True)

fetch_factor=dh.fetch()
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
test2
