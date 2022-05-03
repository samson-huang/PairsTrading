import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Col_zscore(df, n, cap=None, min_periods=1, check_std=False):
    df_mean = df.rolling(window=n,min_periods=min_periods).mean()
    df_std = df.rolling(window=n, min_periods=min_periods).std()
    if check_std:
        df_std = df_std[df_std >= 0.00001]
    target = (df - df_mean) / df_std
    if cap is not None:
        target[target > cap] = cap
        target[target < -cap] = -cap
    return target

def Row_zscore(df, cap=None, check_std=False):
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    if check_std:
        df_std = df_std[df_std >= 0.00001]
    target = df.sub(df_mean, axis=0).div(df_std, axis=0)
    if cap is not None:
        target[target > cap] = cap
        target[target < -cap] = -cap
    return target

def MaxDrawdown(asset_series):
    return asset_series - np.maximum.accumulate(asset_series)

def Sharpe_yearly(pnl_series):
    return (np.sqrt(250) * pnl_series.mean()) / pnl_series.std()

def AnnualReturn(pos_df, pnl_series, alpha_type):
    temp_pnl = (1+pnl_series).prod()
    if alpha_type == 'ls_alpha':
        temp_pos = pos_df.abs().sum().sum() / 2
    else:
        temp_pos = pos_df.abs().sum().sum()
    if temp_pos == 0:
        return .0
    else:
        return round(temp_pnl ** (250 / temp_pos) - 1,2)

def IC(signal, pct_n, min_valids=None, lag=0):
    signal = signal.shift(lag)
    corr_df = signal.corrwith(pct_n, axis=1,method='spearman').dropna()
    if min_valids is not None:
        signal_valid = signal.count(axis=1)
        signal_valid[signal_valid < min_valids] = np.nan
        signal_valid[signal_valid >= min_valids] = 1
        corr_signal = corr_df * signal_valid
    else:
        corr_signal = corr_df
    return corr_signal

def IR(signal, pct_n, min_valids=None, lag=0):
    corr_signal = IC(signal, pct_n, min_valids, lag)
    ic_mean = corr_signal.mean()
    ic_std = corr_signal.std()
    ir = ic_mean / ic_std
    return ir, corr_signal

# def to_weighted_position(selected_df,weights_df = None):
#     if weights_df is None:
#         weights_df = pd.DataFrame().reindex_like(selected_df).fillna(1)
#
#     weights_df = weights_df.reindex_like(selected_df)
#     selected_weights_df = selected_df * weights_df
#     weighted_position_df = selected_weights_df.div(selected_weights_df.sum(axis=1), axis=0)
#     return weighted_position_df

def to_final_position(factor_score, forbid_day):
    '''
    factor_score:DataFrame,可以是因子值，也可以是根据因子值排序选出来的初始仓位矩阵
    forbid_day:DataFrame,是否可交易（由ST股、停盘相乘得到），1代表该股票该日可以交易，不可交易则是NaN

    return:
        pos_fin：DataFrame,最终仓位
    '''

    #因为因子df中，index为x的数值是用x日收盘后更新的数据计算的，所以x日不能交易，需要等到下一天交易，所以要shift（1）
    pos_fin = factor_score.shift(1).replace(np.nan, 0) * forbid_day
    #这里的ffill的效果是，若某只需要交易的股票在调仓日无法交易，那么经过上一个步骤后，其对应位置是nan，而ffill就是让其先继承前一个交易日的仓位
    pos_fin = pos_fin.ffill()
    return pos_fin

def calc_daily_pnl(factor_df, univ_data, rtn_df, idx_rtn,forbid_days,method):
    '''
    :param factor_df:   因子/仓位矩阵
    :param univ_data:   股票池矩阵（如沪深300成分股、中证500成分股等等)
    :param idx_rtn:  指数rtn序列
    :param forbid_days:  合法交易矩阵
    :param rtn_df:    股票rtn矩阵
    :param method_func:   feature/factor/ls_alpha/hg_alpha

    :return:  仓位矩阵+每日仓位收益率序列
    '''
    factor_sel = factor_df.copy()
    factor_sel = factor_sel.reindex_like(univ_data)*univ_data
    forbid_days = forbid_days.reindex_like(factor_sel)
    return_df = rtn_df.reindex_like(factor_sel)
    if method == 'feature' or method == 'factor':
        factor_z = Row_zscore(factor_sel, cap=4.5)
        pos_final = to_final_position(factor_z, forbid_days)
        daily_pnl_final = (pos_final.shift(1) * return_df).sum(axis=1)
        return pos_final,daily_pnl_final
    elif method == 'ls_alpha':
        pos_final = to_final_position(factor_sel, forbid_days)
        daily_pnl_final = (pos_final.shift(1) * return_df).sum(axis=1)
        return pos_final,daily_pnl_final
    elif method == 'hg_alpha':
        pos_final = to_final_position(factor_sel, forbid_days)
        daily_pnl_final = (pos_final.shift(1) * return_df).sum(axis=1) - idx_rtn
        return pos_final,daily_pnl_final

def factor_group(factor_df,forb_day,rtn_df,idx_rtn,univ_data,split_pct_ls):
    '''
    分组回测
    '''
    factor_df = factor_df.reindex_like(univ_data)*univ_data
    factor_score = factor_df
    factor_rank_pct = factor_score.rank(ascending=False, pct=True, axis=1)

    annual_rtn_ls = list()

    plt.figure(figsize=(12, 6))
    for split_pct in split_pct_ls:
        pos_selected = factor_score[(factor_rank_pct > split_pct[0])&(factor_rank_pct <= split_pct[1])]
        pos_selected = pos_selected.where(pd.isnull(pos_selected), 1)
        pos = pos_selected.div(pos_selected.sum(axis=1), axis=0)

        pos = to_final_position(pos, forb_day).reindex(factor_df.index)
        daily_rtn = (pos.shift(1) * rtn_df).sum(axis=1).reindex(factor_df.index)
        annual_rtn = AnnualReturn(pos,daily_rtn,'factor')
        annual_rtn_ls.append(annual_rtn)
        plt.plot((daily_rtn+1).cumprod(), label=str(split_pct))

    plt.title('all factor group backtest return',fontsize = 14)
    plt.legend()
    plt.grid()
    plt.show()

    xticks = range(len(split_pct_ls))
    plt.figure(figsize=(12, 6))
    p = plt.subplot(111)
    p.bar(x = xticks,height = annual_rtn_ls)
    p.set_xticks(xticks)
    p.set_xticklabels([x[1]*10 for x in split_pct_ls])
    plt.title('factor group annual return',fontsize = 14)
    plt.grid()
    plt.show()



def factor_stats(
        factor_df=None,
        chg_n=1,#调仓时间间隔
        univ_data=None,
        rtn_df=None,
        idx_rtn=None,
        forbid_days = None,
        method='factor',#factor\ls_alpha\hg_alpha
        group_split_ls=[(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
):


    if method=='factor':
        #         plt.figure(figsize=(16, 12))
        plt.figure(figsize=(12, 6))


        pos_final,daily_pnl = calc_daily_pnl(factor_df, univ_data, rtn_df, idx_rtn,forbid_days,method)
        plt.plot(daily_pnl.cumsum())
        plt.title('all factor row_Zscore position return',fontsize = 14)
        plt.grid(1)
        plt.show()

        factor_group(
            factor_df,
            forbid_days,
            rtn_df,
            idx_rtn,
            univ_data,
            split_pct_ls=group_split_ls
        )

        pct_n = rtn_df.rolling(window=chg_n).sum()
        ir,IC_series = IR(factor_df, pct_n, lag=chg_n)
        plt.figure(figsize=(12, 6))
        plt.plot(IC_series.cumsum(),label=f'IR:{round(ir,2)},IC_mean:{round(IC_series.mean(),2)}')
        plt.title('IC cumsum',fontsize = 14)
        plt.legend()
        plt.grid(1)
        plt.show()


    else:
        plt.figure(figsize=(16, 6))
        p1 = plt.subplot(111)
        pos = factor_df.reindex(factor_df.index[::chg_n])#初步仓位，没有剔除St股票 也没有shift
        pos = pos.reindex(factor_df.index).ffill()
        pos_final,daily_pnl = calc_daily_pnl(pos, univ_data, rtn_df, idx_rtn,forbid_days,method)
        sharpe = round(Sharpe_yearly(daily_pnl),2)
        max_drawdown = round(MaxDrawdown((daily_pnl+1).cumprod()),2)
        annual_return = round(AnnualReturn(pos_final,daily_pnl,method),2)
        p1.plot(daily_pnl.cumsum(),label=f'SP:{sharpe},MD:{max_drawdown.min()},AR:{annual_return}')
        p1.set_title('selected position return')
        p1.grid(1)
        p1.legend()
    plt.show()
