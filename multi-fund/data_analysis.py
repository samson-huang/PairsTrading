import qlib
from qlib.constant import REG_CN

from qlib.utils import init_instance_by_config

from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)


# 配置数据
train_period = ("2019-01-01", "2021-12-31")
valid_period = ("2022-01-01", "2022-12-31")
test_period = ("2023-01-01", "2023-08-04")



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


