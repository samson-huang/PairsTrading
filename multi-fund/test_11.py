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
from qlib.contrib.report import analysis_model, analysis_position
from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)


if __name__ == '__main__':
    ba_rid='3c34e68ee2be43f29c00473c5a5e6628'

    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")

    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")

    CSI300_BENCH = "SZ160706"
    STRATEGY_CONFIG = {
        "topk": 5,
        "n_drop": 5,
        # pred_score, pd.Series
        "signal": pred_df,
    }

    strategy_obj = NewTopkDropoutStrategy(**STRATEGY_CONFIG)
    account_obj = 1000000
    report_normal, positions_normal = backtest_daily(
        start_time="2023-01-01", end_time="2023-08-24", strategy=strategy_obj,
        account=account_obj,benchmark="SZ160706"
    )
    analysis = dict()
    # default frequency will be daily (i.e. "day")
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])

    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    pprint(analysis_df)

    #df_signal = recorder.load_object('pred.pkl')
    #df_signal.tail()
    #df_signal.loc[['2023-01-03'], :].sort_values(by='score', ascending=False)

    # 对每个datatime分组,排序取前5名
    top5 = pred_df.groupby(level=0)['score'].nlargest(5)