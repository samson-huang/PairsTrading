# 引入本地库
import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//multi-factor//个股动量效应的识别及球队硬币因子//")


from typing import List, Tuple

import empyrical as ep
import pandas as pd
import qlib
from FactorZoo import SportBettingsFactor, VolatilityMomentum
from src.build_factor import get_factor_data_and_forward_return
from src.factor_analyze import get_factor_describe, get_factor_group_returns
from src.plotting import plot_cumulativeline_from_dataframe

qlib.init(region="cn")




all_data: pd.DataFrame = get_factor_data_and_forward_return(
    SportBettingsFactor,
    window=20,
    periods=1,
    general_names=["interday", "intraday", "overnight"],
)