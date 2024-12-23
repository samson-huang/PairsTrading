﻿# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 16:56:01 2022

@author: admin
"""

# 使用ts
import pandas as pd
import numpy as np
import tushare as ts
import json
import datetime
from dateutil.parser import parse
from sklearn.pipeline import Pipeline
from Hugos_tools.Approximation import (Approximation, Mask_dir_peak_valley,
                                          Except_dir, Mask_status_peak_valley,
                                          Relative_values)
# 请根据自己的情况填写ts的token
#setting = json.load(open('C:\config\config.json'))
######################################### tuhsare自动延迟下载 ####################################################
# tuhsare自动延迟下载，防止频繁调取数据是报错
import time
import logging
import logging.handlers


class TuShare:
    """tushare服务接口自动重试封装类，能够在接口超时情况下自动等待1秒然后再次发起请求，
    无限次重试下去，直到返回结果或者达到最大重试次数。
    """

    def __init__(self, token, logger=None, max_retry=0):
        """构造函数，token：tushare的token；logger：日志对象，可以不传；
        max_retry：最大重试次数，默认为0意为无限重试，建议用10以上100以内。"""
        self.token = token
        if not logger:
            logger = logging.getLogger('TuShare')
            # CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s %(name)s %(pathname)s:%(lineno)d %(funcName)s %(levelname)s %(message)s'
            )
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)
        self.logger = logger
        self.max_retry = max_retry
        ts.set_token(token)
        self.pro = ts.pro_api()

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            i = 0
            while True:
                try:
                    if name == 'pro_bar':
                        m = getattr(ts, name, None)
                    else:
                        m = getattr(self.pro, name, None)
                    if m is None:
                        self.logger.error('Attribute %s does not exist.', name)
                        return None
                    else:
                        return m(*args, **kwargs)
                except (Exception):
                    if self.max_retry > 0 and i >= self.max_retry:
                        raise
                    self.logger.exception(
                        'TuShare exec %s failed, args:%s, kwargs:%s, try again.',
                        name, args, kwargs)
                    time.sleep(5)
                i += 1

        return wrapper


#my_pro = TuShare(setting['token'], max_retry=60)  # 初始化ts

################################################################################
######################################### 公用 ####################################################
# 绕过查询限制
def distributed_query(query_func_name,
                      symbol,
                      start_date,
                      end_date,
                      fields,
                      limit=3000):
    n_symbols = len(symbol.split(','))
    #dates = query_trade_dates(start_date, end_date)
    dates = GetEveryDay(start_date,end_date)
    n_days = len(dates)

    if n_symbols * n_days > limit:
        n = limit // n_symbols

        df_list = []
        i = 0
        pos1, pos2 = n * i, n * (i + 1) - 1

        while pos2 < n_days:
            df = query_func_name(
                ts_code=symbol,
                start_date=dates[pos1],
                end_date=dates[pos2],
                fields=fields)
            df_list.append(df)
            i += 1
            pos1, pos2 = n * i, n * (i + 1) - 1
        if pos1 < n_days:
            df = query_func_name(
                ts_code=symbol,
                start_date=dates[pos1],
                end_date=dates[-1],
                fields=fields)
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
    else:
        df = query_func_name(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date,
            fields=fields)
    return df

###################################################################
def distributed_other_query(query_func_name,
                      start_date,
                      end_date,
                      limit=300):
                      	
    dates = GetEveryDay(start_date,end)
    n_days = len(dates)

    if  n_days > limit:
        n = limit 

        df_list = []
        i = 0
        pos1, pos2 = n * i, n * (i + 1) - 1

        while pos2 < n_days:
            df = query_func_name(
                start_date=dates[pos1],
                end_date=dates[pos2])
            df_list.append(df)
            i += 1
            pos1, pos2 = n * i, n * (i + 1) - 1
        if pos1 < n_days:
            df = query_func_name(
                start_date=dates[pos1],
                end_date=dates[-1])
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
    else:
        df = query_func_name(
            start_date=start_date,
            end_date=end_date)
    return df
    
    
# ts的日历需要处理一下才会返回成交日列表
## 减少ts调用 改用jq的数据....
#df = my_pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)

def query_trade_dates(df:list,start_date: str, end_date: str) -> list:    
    dates = df.query('is_open==1')['cal_date'].values.tolist()
    return  dates
    #return get_trade_days(start_date, end_date).tolist()
    
####################################################################
def GetEveryDay(begin_date,end_date):
#获得两日期间的日期列表
    global date_list
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date,"%Y%m%d")
    end_date = datetime.datetime.strptime(end_date,"%Y%m%d")
    while begin_date <= end_date:
          date_str = begin_date.strftime("%Y%m%d")
          date_list.append(date_str)
          begin_date += datetime.timedelta(days=1)
    #print('日期列表已形成')
    return date_list

#################################
def foundation_summary(back_testing):

        back_df = back_testing

        index_name = '年化收益率,累计收益率,夏普比率,最大回撤,持仓总天数,交易次数,平均持仓天数,获利天数, \
        亏损天数,胜率(按天),平均盈利率(按天),平均亏损率(按天),平均盈亏比(按天),盈利次数,亏损次数, \
        单次最大盈利,单次最大亏损,胜率(按此),平均盈利率(按次),平均亏损率(按次),平均盈亏比(按次)'.split(
            ',')

        # 寻找标列
        mark_list = [x for x in back_df.columns if x.split('_')[-1] == 'MARK']

        temp = []
        mark_size = len(mark_list)  # 列数

        if mark_size > 1:

            for m in mark_list:

                df = pd.DataFrame(risk_indicator(
                    back_df, m), index=index_name)
                temp.append(df)

            return pd.concat(temp, axis=1)

        else:

            return pd.DataFrame(risk_indicator(back_df, m), index=index_name)

    # 计算风险指标

def risk_indicator(x_df, mark_col):
    	
        '''
        传入经back_testing

        '''
        df = x_df.copy()

        summary_dic = {}

        # 格式化数据
        def format_x(x):
            return '{:.2%}'.format(x)

        # 获取回测数据
        df['pct_chg'] = df['pct_chg']/100
        df['NEXT_RET'] = df['pct_chg'].shift(-1)

        NOT_NAN_RET = df['NEXT_RET'].dropna()*df[mark_col]
        RET = df['NEXT_RET']*df[mark_col]

        CUM_RET = (1+RET).cumprod()  # series

        # 计算年化收益率
        annual_ret = CUM_RET.dropna()[-1]**(250/len(NOT_NAN_RET)) - 1

        # 计算累计收益率
        cum_ret_rate = CUM_RET.dropna()[-1] - 1

        # 最大回撤
        max_nv = np.maximum.accumulate(np.nan_to_num(CUM_RET))
        mdd = -np.min(CUM_RET / max_nv - 1)

        # 夏普
        sharpe_ratio = np.mean(NOT_NAN_RET) / \
            np.nanstd(NOT_NAN_RET, ddof=1)*np.sqrt(250)

        # 盈利次数
        temp_df = df.copy()

        diff = temp_df[mark_col] != temp_df[mark_col].shift(1)
        temp_df[mark_col+'_diff'] = diff.cumsum()
        cond = temp_df[mark_col] == 1
        # 每次开仓的收益率情况
        temp_df = temp_df[cond].groupby(mark_col+'_diff')['NEXT_RET'].sum()

        # 标记买入卖出时点
        mark = df[mark_col]
        pre_mark = np.nan_to_num(df[mark_col].shift(-1))
        # 买入时点
        trade = (mark == 1) & (pre_mark < mark)

        # 交易次数
        trade_count = len(temp_df)

        # 持仓总天数
        total = np.sum(mark)

        # 平均持仓天数
        mean_hold = total/trade_count
        # 获利天数
        win = np.sum(np.where(RET > 0, 1, 0))
        # 亏损天数
        lose = np.sum(np.where(RET < 0, 1, 0))
        # 胜率
        win_ratio = win/total
        # 平均盈利率（天）
        mean_win_ratio = np.sum(np.where(RET > 0, RET, 0))/win
        # 平均亏损率（天）
        mean_lose_ratio = np.sum(np.where(RET < 0, RET, 0))/lose
        # 盈亏比(天)
        win_lose = win/lose

        # 盈利次数
        win_count = np.sum(np.where(temp_df > 0, 1, 0))
        # 亏损次数
        lose_count = np.sum(np.where(temp_df < 0, 1, 0))
        # 单次最大盈利
        max_win = np.max(temp_df)
        # 单次最大亏损
        max_lose = np.min(temp_df)
        # 胜率
        win_rat = win_count/len(temp_df)
        # 平均盈利率（次）
        mean_win = np.sum(np.where(temp_df > 0, temp_df, 0))/len(temp_df)
        # 平均亏损率（天）
        mean_lose = np.sum(np.where(temp_df < 0, temp_df, 0))/len(temp_df)
        # 盈亏比(次)
        mean_wine_lose = win_count/lose_count

        summary_dic[mark_col] = [format_x(annual_ret), format_x(cum_ret_rate), sharpe_ratio, format_x(
            mdd), total, trade_count, mean_hold, win, lose, format_x(win_ratio), format_x(mean_win_ratio),
            format_x(mean_lose_ratio), win_lose, win_count, lose_count, format_x(
                max_win), format_x(max_lose),
            format_x(win_rat), format_x(mean_win), format_x(mean_lose), mean_wine_lose]

        return summary_dic

##############################
###############################################
def get_clf_wave(price: pd.DataFrame,
                 rate: float,
                 method: str,
                 except_dir: bool = True,
                 show_tmp: bool = False,
                 dropna: bool = True) -> pd.DataFrame:
    
    
    if except_dir:
        
        # 修正
        perpare_data = Pipeline([('approximation', Approximation(rate, method)),
                ('mask_dir_peak_valley',Mask_status_peak_valley('dir')),
                ('except', Except_dir('dir')),
                ('mask_status_peak_valley', Mask_dir_peak_valley('status'))
                ])
    else:
        
       # 普通
        perpare_data = Pipeline([('approximation', Approximation(rate, method)),
                ('mask_dir_peak_valley',Mask_dir_peak_valley('dir')),
                ('mask_status_peak_valley', Mask_status_peak_valley('dir'))])
        
   

    return perpare_data.fit_transform(price) 