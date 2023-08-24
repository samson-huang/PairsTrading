import qlib
from qlib.constant import REG_CN

from qlib.utils import init_instance_by_config
from pprint import pprint
import pandas as pd
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import NewTopkDropoutStrategy

from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)


if __name__ == '__main__':
    ba_rid='d0f79b75f77548b19d5e4d67fe1aa4f9'

    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")

    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")

    CSI300_BENCH = "SH000300"
    STRATEGY_CONFIG = {
        "topk": 5,
        "n_drop": 0,
        # pred_score, pd.Series
        "signal": pred_df,
    }

    strategy_obj = NewTopkDropoutStrategy(**STRATEGY_CONFIG)
    report_normal, positions_normal = backtest_daily(
        start_time="2023-01-01", end_time="2023-08-04", strategy=strategy_obj
    )
    analysis = dict()
    # default frequency will be daily (i.e. "day")
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])

    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    pprint(analysis_df)