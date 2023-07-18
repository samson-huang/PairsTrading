# 引入本地库
import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//multi-factor//筹码因子//")


import empyrical as ep
import pandas as pd

from pathlib import Path
import qlib
from qlib.data import D
from qlib.utils import init_instance_by_config
from typing import List, Tuple, Dict
from scr.turnover_coefficient_ops import ARC, VRC, SRC, KRC
from scr.cyq_ops import (
    CYQK_C_T,
    CYQK_C_U,
    CYQK_C_TN,
    ASR_T,
    ASR_U,
    ASR_TN,
    CKDW_T,
    CKDW_U,
    CKDW_TN,
    PRP_T,
    PRP_U,
    PRP_TN,
)
from scr.factor_analyze import clean_factor_data, get_factor_group_returns
from scr.qlib_workflow import run_model, get_dataset_config, get_tsdataset_config
from scr.plotting import plot_dist_chips, model_performance_graph, report_graph

import matplotlib.pyplot as plt

# plt中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt显示负号
plt.rcParams["axes.unicode_minus"] = False

custom_ops: List = [ARC, VRC, SRC, KRC, CYQK_C_T, CYQK_C_U, CYQK_C_TN,
                    ASR_T, ASR_U, ASR_TN, CKDW_T, CKDW_U, CKDW_TN, PRP_T, PRP_U, PRP_TN]
qlib.init(
          region="cn", custom_ops=custom_ops)
###################################
# 参数配置
###################################
# 数据处理器参数配置：整体数据开始结束时间，训练集开始结束时间，股票池
TARIN_PERIODS: Tuple = ("2014-01-01", "2017-12-31")
VALID_PERIODS: Tuple = ("2018-01-01", "2020-12-31")
TEST_PERIODS: Tuple = ("2021-01-01", "2023-02-17")


dataset_config: Dict = get_dataset_config(
    "pool", TARIN_PERIODS, VALID_PERIODS, TEST_PERIODS, "TurnCoeffChips"
)

if Path("factor_data/turnovercoeff_dataset.pkl").exists():
    import pickle

    with open("factor_data/turnovercoeff_dataset.pkl", "rb") as f:
        turncoeff_dataset = pickle.load(f)
else:
    # 实例化数据集，从基础行情数据计算出的包含所有特征（因子）和标签值的数据集。
    turncoeff_dataset = init_instance_by_config(dataset_config)  # 类型DatasetH

    # 保存数据方便后续使用
    turncoeff_dataset.config(dump_all=True, recursive=True)
    turncoeff_dataset.to_pickle(
        path="factor_data/turnovercoeff_dataset.pkl", dump_all=True
    )          
