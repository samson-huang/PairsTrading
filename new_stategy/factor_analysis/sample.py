import sys
#
#path_local = "G://GitHub"
path_local = "C://Users//huangtuo//Documents//GitHub"
sys.path.append(path_local+"//PairsTrading//new_stategy//factor_analysis//")

from my_lib.data_download.data_io import DataReader
from my_lib.factor_evaluate.factor_evaluate import factor_stats
import pandas as pd
import numpy as np

def calc_factor():
	#这里是计算因子的，把因子处理成特定的（di,ii)格式。
    close_df = DataReader.read_dailyMkt('close')
    return close_df.pct_change(20)

#计算因子
factor_df = calc_factor()
factor_df.tail(5)

#股票池
univ_a = DataReader.read_IdxWeight('399300.SZ')#沪深300
univ_a = univ_a.where(pd.isnull(univ_a),1)
univ_a.tail(5)


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

#指数对冲
#排序选股，构建仓位
factor_rank_pct = factor_df.rank(ascending=False, pct=True, axis=1)
factor_selected = factor_df[factor_rank_pct>0.8]
factor_selected = factor_selected.where(pd.isnull(factor_selected), 1)
pos = factor_selected.div(factor_selected.sum(axis=1), axis=0)
pos = pos.fillna(0)#重要，否则ffill会出错


factor_stats(
    factor_df = pos,
    chg_n=20,
    univ_data=univ_a,
    rtn_df=rtn_df,
    idx_rtn=idx_rtn.replace(np.inf,np.nan).replace(-np.inf,np.nan),
    forbid_days=forb_days,
    method='hg_alpha',#factor\ls_alpha\hg_alpha
)

#多空对冲
factor_df = factor_df.reindex_like(univ_a)*univ_a
factor_rank_pct = factor_df.rank(ascending=False, pct=True, axis=1)

#多头仓位
factor_selected = factor_df[factor_rank_pct>0.8]
factor_selected = factor_selected.where(pd.isnull(factor_selected), 1)
pos_long = factor_selected.div(factor_selected.sum(axis=1), axis=0).fillna(0)
#空头仓位
factor_selected = factor_df[factor_rank_pct<0.2]
factor_selected = factor_selected.where(pd.isnull(factor_selected), 1)
pos_short = factor_selected.div(factor_selected.sum(axis=1), axis=0).fillna(0)

factor_stats(
    factor_df = pos_long.fillna(0) - pos_short.fillna(0),
    chg_n=20,
    univ_data=univ_a,
    rtn_df=rtn_df,
    idx_rtn=idx_rtn.replace(np.inf,np.nan).replace(-np.inf,np.nan),
    forbid_days=forb_days,
    method='ls_alpha',#factor\ls_alpha\hg_alpha
)

#

#转换城alphalens-example方式
test_df=rtn_df
test_df=test_df.reset_index()
test_df=test_df.melt(id_vars=['trade_date'],var_name='ts_code',value_name='pct_chg')

assets = test_df.set_index([test_df['trade_date'], test_df['ts_code']], drop=True)
assets = assets.drop(['trade_date','ts_code'],axis=1)
assets.tail()




