local_path="c://Users//huangtuo//Documents//"
import sys
sys.path.append(local_path+"GitHub//PairsTrading//new_stategy//factor_analysis//")


from my_lib.data_download.data_io import DataReader
from my_lib.factor_evaluate.factor_evaluate import factor_stats
import pandas as pd
import numpy as np

def calc_factor():
	#这里是计算因子的，把因子处理成特定的（di,ii)格式。
    close_df = DataReader.read_dailyMkt('close')
    return close_df.pct_change(20)

factor_df = calc_factor()
factor_df.tail(5)
#股票池
univ_a = DataReader.read_IdxWeight('399300.SZ')#沪深300
univ_a = univ_a.where(pd.isnull(univ_a),1)
univ_a

#st股、停牌、涨跌停
ST_valid = DataReader.read_ST_valid()
suspend_valid = DataReader.read_suspend_valid()
limit_valid = DataReader.read_limit_valid()
forb_days = ST_valid*suspend_valid*limit_valid
forb_days.tail(5)

#每日收益率矩阵
rtn_df = DataReader.read_dailyRtn()
rtn_df.tail(5)

#因子测试与回测
#原始因子
idx_rtn = DataReader.read_index_dailyRtn('399300.SZ')#指数收益序列
factor_stats(
    factor_df=factor_df,#因子或者仓位矩阵
    chg_n=20,#调仓周期
    univ_data=univ_a,#股票池
    rtn_df=rtn_df,#股票每日收益矩阵
    idx_rtn=idx_rtn,#因子收益序列（用来hg对冲）
    forbid_days=forb_days,#st*suspend*limit
    method='factor',#factor\ls_alpha\hg_alpha（原始因子、指数对冲、多空对冲）
    group_split_ls=[(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]#分组回测参数
)