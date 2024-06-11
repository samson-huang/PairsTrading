import qlib
from qlib.constant import REG_CN
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

from qlib.data import D
import talib
from typing import List, Tuple, Dict
# 配置数据
#train_period = ("2019-01-01", "2021-12-31")
#valid_period = ("2022-01-01", "2022-12-31")
test_period = ("2021-01-01", "2024-05-14")

market = "filter_fund"
benchmark = "SZ160706"
# 获取test时段的行情原始数据
stockpool: List = D.instruments(market=market)
raw_data: pd.DataFrame = D.features(
    stockpool,
    fields=["$open", "$high", "$low", "$close", "$volume"],
    start_time=test_period[0],
    end_time=test_period[1],
)

for col in ['$open', '$high', '$low', '$close', '$volume']:
    raw_data[col] = raw_data[col].fillna(method='ffill')

raw_data['$SMA'] = raw_data.groupby(level='instrument')['$close'].apply(lambda x: talib.SMA(x, timeperiod=20))



