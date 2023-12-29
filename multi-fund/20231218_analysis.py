import qlib
from qlib.constant import REG_CN

from qlib.utils import init_instance_by_config

from qlib.workflow import R
from qlib.utils import flatten_dict
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord,SigAnaRecord
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

#######################################################################################
#####行业有效量价因子与行业轮动策略ETF后续结果分析代码借用#############
#####嫁接自己得fund判断########################
#####################################################################################
import pandas as pd
from typing import Dict, List, Tuple
#import matplotlib.pyplot as plt

#预测结果查询
ba_rid='e5b64c6cf42e44a388c523624f71a5b9'
#使用os.chdir()修改当前工作目录
R.set_uri("file:C:\\Users\\huangtuo\\mlruns")
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

#使用Backtrader根据预测值回测
import sys
from qlib.data import D # 基础行情数据服务的对象
sys.path.append("C:/Local_library/")

#pred_label_df 为前面生成的图片
from scr.plotting import model_performance_graph, report_graph

'''
report_normal_1day_df: pd.DataFrame = predict_recorder.load_object(
    "portfolio_analysis/report_normal_1day.pkl")
report_graph(report_normal_1day_df)
'''
from hugos_toolkit_old.BackTestTemplate import Top5Strategy,get_backtesting,AddSignalData
from hugos_toolkit_old.BackTestReport.tear import analysis_rets


def get_backtest_data(
    pred_df: pd.DataFrame, start_time: str, end_time: str,market='market'
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
        ["SZ160706"],
        fields=["$close"],
        start_time=start_time,
        end_time=end_time,
    ).reset_index(level=0, drop=True)

    return data, benchmark
if __name__ == '__main__':

    test_period = ("2023-01-01", "2023-08-24")
    market = "filter_fund"
    data,benchmark = get_backtest_data(pred_df,test_period[0],test_period[1],market=market)

    benchmark_ret:pd.Series = benchmark['$close'].pct_change()

    bt_result = get_backtesting(
        data,
        strategy=Top5Strategy,
        mulit_add_data=True,
        feedsfunc=AddSignalData,
        strategy_params={"selnum": 5, "pre": 0.05,'ascending':False,'show_log':False},
    )