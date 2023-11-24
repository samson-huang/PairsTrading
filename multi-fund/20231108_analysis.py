

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
benchmark = "SH000300"

###################################
# train model
###################################
data_handler_config = {
    "start_time": train_period[0],
    "end_time": test_period[1],
    "fit_start_time": train_period[0],
    "fit_end_time": train_period[1],
    "instruments": market,
}

task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": train_period,
                "valid": valid_period,
                "test": test_period,
            },
        },
    },
}


# model initiaiton
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# start exp to train model
with R.start(experiment_name="train_model"):
    R.log_params(**flatten_dict(task))
    model.fit(dataset)
    R.save_objects(trained_model=model)
    rid = R.get_recorder().id

###################################
# prediction, backtest & analysis
###################################
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "model": model,
            "dataset": dataset,
            "topk": 10,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": test_period[0],
        "end_time": test_period[1],
        "account": 100000000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

# backtest and analysis
with R.start(experiment_name="backtest_analysis"):
    recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
    model = recorder.load_object("trained_model")

    # prediction
    recorder = R.get_recorder()
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()

    # 生成预测结果分析文件，在artifacts\sig_analysis 目录生成ic.pkl,ric.pkl文件
    sigAna_rec = SigAnaRecord(recorder) # 信号分析记录器
    sigAna_rec.generate()


from qlib.contrib.report import analysis_model, analysis_position

recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")
print(recorder)
pred_df = recorder.load_object("pred.pkl")
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")

#analysis position
#report
analysis_position.report_graph(report_normal_df)

#analysis model
label_df = dataset.prepare("test", col_set="label")
label_df.columns = ["label"]





pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
analysis_position.score_ic_graph(pred_label)

#model performance
analysis_model.model_performance_graph(pred_label)

# 回测报告（净值，交易成本等）
recorder.load_object('portfolio_analysis/report_normal_1day.pkl')

#当我们看TopKDropout的源码时，我们发现TopKDropout是继承自BaseSignalStrategy。BaseSignalStrategy也是BaseStrategy的一个子类，
# 它与BaseStrategy最大的区别是属性中增加了signal，signal就是某个时间对某个股票的预测，
# 一般是机器学习模型在测试集上的预测结果。比如上面预测结果pred就是signal：
df_signal = recorder.load_object('pred.pkl')
df_signal.tail()
df_signal.loc[['2023-08-04'], :].sort_values(by='score', ascending=False)

# 准备训练数据
dataset.prepare('train')

# 准备测试数据
dataset.prepare('test')

#使用Backtrader根据预测值回测
import sys
from qlib.data import D # 基础行情数据服务的对象
sys.path.append("C:/Local_library/")

def get_backtest_data(
    pred_df: pd.DataFrame, start_time: str, end_time: str,market='market'):

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
        ["SH000300"],
        fields=["$close"],
        start_time=start_time,
        end_time=end_time,
    ).reset_index(level=0, drop=True)

    return data, benchmark

data,benchmark = get_backtest_data(pred_df,test_period[0],test_period[1],market = "20230820_fund")

benchmark_ret:pd.Series = benchmark['$close'].pct_change()

#######################################################################################
#####行业有效量价因子与行业轮动策略ETF后续结果分析代码借用#############
#####嫁接自己得fund判断########################
#####################################################################################
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//multi-fund//foundation_tools//")

#预测结果查询
predict_recorder= R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")
# 这个pkl文件记录的是测试集未经数据预处理的原始标签值
label_df = predict_recorder.load_object("label.pkl")
# 修改列名LABEL0为label 这个label其实就是下一期得收益率
label_df.columns = ['label']
pred_df = predict_recorder.load_object("pred.pkl") # 加载测试集预测结果到dataframe

print('label_df', label_df) # 预处理后的测试集标签值
print('pred_df', pred_df) # 测试集对标签的预测值，score就是预测值

#IC,Rank IC查询
ic_df = predict_recorder.load_object("sig_analysis/ic.pkl")

ric_df = predict_recorder.load_object("sig_analysis/ric.pkl")

# 所有绩效指标
print("list_metrics", predict_recorder.list_metrics())
# IC均值：每日IC的均值，一般认为|IC|>0.03说明因子有效，注意 -0.05也认为有预测效能，说明负相关显著
print("IC", predict_recorder.list_metrics()["IC"])
# IC信息率：平均IC/每日IC标准差,也就是方差标准化后的ic均值，一般而言，认为|ICIR|>0.6,因子的稳定性合格
print("ICIR", predict_recorder.list_metrics()["ICIR"])
# 排序IC均值，作用类似IC
print("Rank IC", predict_recorder.list_metrics()["Rank IC"])
# 排序IC信息率，作用类似ICIR# 此图用于评价因子单调性，组1是因子值最高的一组，组5是因子值最低的一组。
print("Rank ICIR", predict_recorder.list_metrics()["Rank ICIR"])

# 创建测试集"预测"和“标签”对照表
pred_label_df = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(label_df.index)
pred_label_df.head()

#查看IC及分组收益情况
from scr.plotting import plot_qlib_factor_dist

#pip install C:\20231108\TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
plot_qlib_factor_dist(pred_label_df,no_raise=True)

#需要先调用train_model.py生成模型
with R.start():
    recorder_158 = R.get_recorder(experiment_id='429618400927874750', recorder_id='be8bd353a9524bbb83d659e3b050bde7')
    model_158 = recorder_158.load_object("trained_model")


#使用Backtrader根据预测值回测
from hugos_toolkit.BackTestTemplate import StockSelectStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
from hugos_toolkit.BackTestReport.tear import analysis_rets
#使用Backtrader根据预测值回测










