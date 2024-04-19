import qlib
from qlib.constant import REG_CN
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from datetime import datetime
#ranked_data.to_csv("c:\\temp\\ranked_data_20240221.csv")
ranked_data = pd.read_csv('c:\\temp\\ranked_data_all.csv', parse_dates=['datetime'], index_col='datetime')
#导入hugos_toolkit库需要指定目录
import sys
import os
local_path = os.getcwd()
sys.path.append(local_path+'\\Local_library\\')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets
from hugos_toolkit.BackTestTemplate import MultiTestStrategy

class AddSignalData(bt.feeds.PandasData):
    """用于加载回测用数据

    添加信号数据
    """
    lines = ("code",)
    lines = ("rank",)

    params = (("rank", -1),("code", -2),)


#主函数
if __name__ == '__main__':
    data: pd.DataFrame = ranked_data
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000)
    if (begin_dt is None) or (end_dt is None):
        begin_dt = data.index.min()
        end_dt = data.index.max()
    datafeed = AddSignalData(dataname=data, fromdate=begin_dt, todate=end_dt)
    cerebro.adddata(datafeed, name=MultiTestStrategy)

