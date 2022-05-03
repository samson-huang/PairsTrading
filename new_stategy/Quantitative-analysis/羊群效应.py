import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
from jqdatajdk import *
#from jqfactor import *
import datetime as dt
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from IPython.core.display import HTML

# 设置字体 用来正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('ggplot')

# 忽略报错
import warnings
warnings.filterwarnings("ignore")


start = '20140101'
end = '20190831'
interval = 22

import sys 
sys.path.append("G://GitHub//PairsTrading//new_stategy//foundation_tools//") 
import foundation_tushare 
import json

# 请根据自己的情况填写ts的token
setting = json.load(open('C:\config\config.json'))
my_pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)

############tushare替换函数##############################
def TdaysOffset(end_date: str, count: int) -> int:
    '''
    end_date:为基准日期
    count:为正则后推，负为前推
    -----------
    return datetime.date
    '''
    df = my_pro.trade_cal(exchange='SSE')
    test_2=df[df['cal_date']==end_date].index[0]
    if count > 0:
        df1=df[test_2:]
        df1=df1[df1['is_open']==1]
        df1=df1.reset_index(drop=True)
        return df1[df1.index==count]['cal_date'].values[0]
    elif count < 0:
        df1=df[:test_2+1]
        df1=df1[df1['is_open']==1]
        df1=df1.reset_index(drop=True)
        count_df1=len(df1)+count
        return df1[df1.index==count_df1]['cal_date'].values[0]
    else:

        raise ValueError('别闹！')    

def GetEveryDay(begin_date,end_date):
#获得两日期间的日期列表
    global date_list
    date_list = []
    begin_date = datetime.strptime(begin_date,"%Y%m%d")
    end_date = datetime.strptime(end_date,"%Y%m%d")
    while begin_date <= end_date:
          date_str = begin_date.strftime("%Y%m%d")
          date_list.append(date_str)
          begin_date += datetime.timedelta(days=1)
    #print('日期列表已形成')
    return date_list
##############数据获取函数###################################
# 1 获取指数成分股
# 过滤ST；过滤上市不足三个月=>filter_now_share；过滤当日停牌股票=>filter_paused_stocks
def filter_index_stocks(index_code, trade_date):
    # 获取成分股列表
    #stocks = get_index_stocks(index_code, date=trade_date)
    stocks = my_pro.index_weight(index_code=index_symbol, date=end_date)
    # 剔除ST股
    st_stocks = get_extras('is_st', stocks, count=1, end_date=trade_date)
    stockList = [stock for stock in st_stocks if not st_stocks[stock][0]]

    # 剔除上市不足三月股票
    stockList = filter_now_share(stockList, trade_date)
    # 剔除当日停牌股票
    stockList = filter_paused_stocks(stockList, trade_date)
    return stockList


# 2 过滤上市不足3月的股票
def filter_now_share(stocks, begin_date, n=3 * 30):
    stockList = []
    # 如果begin_date
    if type(begin_date) == str:
        begin_date = dt.datetime.strptime(begin_date, "%Y-%m-%d").date()
    for stock in stocks:
        start_date = get_security_info(stock).start_date
        if start_date < (begin_date - dt.timedelta(days=n)):
            stockList.append(stock)
    return stockList


# 3 过滤当日停牌股票
def filter_paused_stocks(stockList, begin_date):
    is_paused = get_price(stockList,
                          end_date=begin_date,
                          count=1,
                          fields='paused')['paused'].T
    unsuspened_stocks = is_paused[is_paused.iloc[:, 0] < 1].index.tolist()
    return unsuspened_stocks



# 4 获取基础数据
'''
index_code：为成分股代码
start,end：str 日期
interval为N日收益率
file_name:储存文件的文件名
------------------------
实际运行时数据日期会在start向前推interval日
return dict key['close','smb'] value:df index为date,columns为股票代码
'''


def get_datas(index_code, start, end, interval, file_name):

    # 向前推N日
    begin = TdaysOffset(end_date=start, count=interval)[0]
    # 获取交易日利
    trade_list = GetEveryDay(start_date=begin, end_date=end)

    datas = {}
    stock_df = []
    smb_list = []

    for date in trade_list:
        # 获取股票列表
        stocksList = filter_index_stocks(index_code, date)

        # 获取close数据
        stocks = get_price(stocksList, end_date=date, count=1,
                           fields='close')['close']
        stock_df.append(stocks)

        # 获取smb数据
        q = query(valuation.code, valuation.day,
                  valuation.circulating_market_cap).filter(
                      valuation.code.in_(stocksList))
        smb = get_fundamentals(q, date=date)
        # 单位亿元
        smb = smb.pivot(index='day',
                        columns='code',
                        values='circulating_market_cap')
        smb_list.append(smb)

        print('success', date.strftime('%Y-%m-%d'))
    # 合并数据
    close_df = pd.concat(stock_df)
    smb_df = pd.concat(smb_list)
    datas['close'] = close_df
    datas['smb'] = smb_df
    # 存储数据
    pkl_file = open(file_name, 'wb')
    pickle.dump(datas, pkl_file)
    print('以储存数据：' + file_name)
    return datas


# 5 获取信号
'''
datas_df：成分股数据，index为日期，columns为代码
interval为间隔N日收益率,22为月收益率
file_name:储存文件的文件名
------------------------
实际运行时数据日期会在start向前推interval日
return df
'''


def get_factor(dic, index_code, interval):

    stocks = dic['close']
    smb_df = dic['smb']
    # 获取日期
    begin = min(hs300_datas['close'].index).strftime('%Y-%m-%d')
    end = max(hs300_datas['close'].index).strftime('%Y-%m-%d')

    index_close = get_price(index_code,
                            start_date=begin,
                            end_date=end,
                            fields='close')['close']
    # 成分股
    # N-1实际为22日得收益率
    stock_ret = stocks.pct_change(interval - 1)
    stock_ret = stock_ret[interval - 1:]

    smb_ret = smb_df.pct_change(interval - 1)
    smb_ret = smb_ret[interval - 1:]
    # 指数
    index_Nret = index_close.pct_change(interval - 1)
    index_Nret = index_Nret[interval - 1:]
    # 指数日收益率
    index_ret = index_close.pct_change()
    index_ret = index_ret[interval - 1:]
    # 指数收盘价
    index_close = index_close[interval - 1:]

    # 将指数的一维数据转为与成分股相同的矩阵相减
    ret_diff_arr = stock_ret.values - np.broadcast_to(
        np.expand_dims(index_Nret.values, axis=1), (stock_ret.values.shape))
    smb_diff_arr = smb_ret.values - np.broadcast_to(
        np.expand_dims(index_Nret.values, axis=1), (smb_ret.values.shape))
    # 计算CSAD
    csad_arr = np.nansum(abs(ret_diff_arr), axis=1) / np.count_nonzero(
        ret_diff_arr, axis=1)
    csad_smb = np.nansum(abs(smb_diff_arr), axis=1) / np.count_nonzero(
        smb_diff_arr, axis=1)
    # 收益绝对值
    ret_arr = abs(stock_ret.values)
    smb_arr = abs(smb_ret.values)
    # 收益方
    ret_2_arr = ret_arr**2
    smb_2_arr = smb_arr**2
    # 回归
    temp = []
    for i in range(len(stock_ret)):
        trade_date = stock_ret.index[i]
        X_a = np.nan_to_num(np.c_[ret_arr[i], ret_2_arr[i]])
        Y_a = np.broadcast_to(np.expand_dims(csad_arr[i], axis=0),
                              (len(X_a), 1))
        beta = np.linalg.lstsq(X_a, Y_a)[0][1][0]  # 获取回归系数，最小二乘法

        X_b = np.nan_to_num(np.c_[smb_arr[i], smb_2_arr[i]])
        Y_b = np.broadcast_to(np.expand_dims(csad_smb[i], axis=0),
                              (len(X_b), 1))
        beta_smb = np.linalg.lstsq(X_b, Y_b)[0][1][0]

        r_score = cal_score(beta, index_Nret[i])
        smb_score = cal_score(beta_smb, index_Nret[i])

        temp.append([
            beta, beta_smb, index_Nret[i], index_ret[i], index_close[i],
            beta < 0 and index_Nret[i] > 0, beta < 0 and index_Nret[i] < 0,
            beta_smb < 0 and index_Nret[i] < 0, r_score, smb_score
        ])

    # 列名
    column_name = [
        'r_csad', 'smb_csad', 'index_Nret', '当日涨幅', 'close', 'r_up_singal',
        'r_down_singal', 'smb_singal', 'r_factor', 'smb_factor'
    ]
    # 构建df
    factor_df = pd.DataFrame(temp, columns=column_name, index=stock_ret.index)
    # 存储数据
    #pkl_file=open(file_name,'wb')
    #pickle.dump(factor,pkl_file)
    #print('以储存数据：'+file_name)
    return factor_df


# 5-1 将CSAD和指数收益合成打分
def cal_score(csad, ret):
    if csad < 0 and ret > 0:
        score = csad + ret
    else:
        score=abs(csad)+abs(ret)
    return score
################################################################
###################回测用函数###################################
# 6 回测函数
'''
输入：df index为日期,singal_col为df中含信号得列名
---------------
总体逻辑是有信号则买入持有，无信号则平仓
'''


def back_test(df, singal_col,holding=None, prt=False):
    position = []

    if holding == None:
        for i in range(len(df)):
            factor_value = df[singal_col][i]
            if factor_value:
                position.append(1)
            else:
                position.append(0)
    else:
        count = holding
        for i in range(len(df)):
            factor_value = df[singal_col][i]
            if factor_value:
                position.append(1)
                count = 1
            else:
                if count < holding:
                    count += 1
                    position.append(1)
                else:
                    position.append(0)

    df['position'] = position
    if prt:
        position = np.array(position)
        print('满仓天数：', len(position[position == 1]))
        print('空仓天数：', len(position[position == 0]))
    # 计算收益率
    index_ret = df.close.pct_change().values  #df['当日收益率']
    ret = [0]
    # 确定哪些是开仓位置
    for i in range(len(df) - 1):
        ret.append(index_ret[i + 1] * position[i])
    ret = np.array(ret)
    df['ret'] = ret
    cum_ret = []
    # 计算净值
    for i in range(len(ret)):
        if i == 0:
            cum_ret.append(1 + ret[i])
        else:
            cum_ret.append(cum_ret[-1] * (1 + ret[i]))
    df['cum_ret'] = cum_ret
    return df


# 7 生成回测报告
def summary(df):
    #输出各项指标
    cum_ret = df['cum_ret']
    ret = df['ret']
    # 计算年华收益率
    annual_ret = cum_ret[-1]**(240 / (len(ret) - 5)) - 1
    # 计算累计收益率
    cum_ret_rate = cum_ret[-1] - 1
    # 最大回撤
    max_nv = np.maximum.accumulate(cum_ret)
    mdd = -np.min(cum_ret / max_nv - 1)

    print('年化收益率: {:.2%}'.format(annual_ret))
    print('累计收益率: {:.2%}'.format(cum_ret_rate))
    print('最大回撤: {:.2%}'.format(mdd))
    print('夏普比率：{:.2}'.format(ret.mean() / ret.std() * np.sqrt(240)))

    #作图
    plt.figure(1, figsize=(20, 10))
    plt.title('净值曲线', fontsize=18)
    plt.plot(df.index, cum_ret)
    plt.plot(df.index, df['close'] / df['close'][0])
    plt.legend(['策略净值', '基准净值'], fontsize=15)
    plt.figure(2, figsize=(20, 10))
    plt.title('相对优势', fontsize=18)
    plt.plot(df.index, cum_ret - df['close'] / df['close'][0])
    plt.show()


# 分组回测
'''
df 为分组后的信号数据
group_col 为有分组的列名
holding 为持有天数
'''


def group_back_test(df, group_col, holding=None):

    group_list = df[group_col].unique().tolist()
    group_list.sort()

    ret_dic = {}  # 储存每组回测收益率
    cum_ret_dic = {}  # 储存每组净值
    report = {}  # 储存每组报告数据

    # 获取指数每日收益率
    index_ret = df.close.pct_change().values
    # 获取指数收盘价
    index_close = df.close.values
    index_close = index_close[:-1]
    # 基准净值
    cum_ret_dic['基准净值'] = index_close / index_close[0]

    for group_num in group_list:

        ret = []  # 储存收益率
        cum_ret = []  # 储存净值

        # 标注
        #-----------------------------------
        if holding == None:
            position = np.zeros(len(df))
            mask = df[group_col] == group_num
            position[mask] = 1
        else:
            count = holding
            position = []
            for i in range(len(df)):
                threshold = df[group_col][i]
                if threshold == group_num:
                    position.append(1)
                    count = 1
                else:
                    if count < holding:
                        count += 1
                        position.append(1)
                    else:
                        position.append(0)
        #------------------------------------

        # 计算收益率
        for i in range(len(index_ret) - 1):
            # 取滞后一期得收益
            ret.append(index_ret[i + 1] * position[i])

        ret = np.array(ret)
        winning_count=np.sum(np.where(ret>0,1,0))/np.count_nonzero(position) # 日胜率
        ret_dic[group_num] = ret

        # 计算净值
        for i in range(len(ret)):
            if i == 0:
                cum_ret.append(1 + ret[i])
            else:
                cum_ret.append(cum_ret[-1] * (1 + ret[i]))

        cum_ret_dic[group_num] = cum_ret
        #-------------------关键指标计算----------------------
        # 计算年化收益率
        annual_ret = cum_ret[-1]**(240 / (len(ret) - 5)) - 1
        # 计算累计收益率
        cum_ret_rate = cum_ret[-1] - 1
        # 最大回撤
        max_nv = np.maximum.accumulate(cum_ret)
        mdd = -np.min(cum_ret / max_nv - 1)

        # 储存每组报告数据
        report[group_num] = {
            '满仓天数': np.count_nonzero(position),
            '空仓天数': len(position) - np.count_nonzero(position),
            '日胜率':'{:.2%}'.format(winning_count),
            '年化收益率': '{:.2%}'.format(annual_ret),
            '累计收益率': '{:.2%}'.format(cum_ret_rate),
            '最大回撤': '{:.2%}'.format(mdd),
            '夏普比率': '{:.2}'.format(ret.mean() / ret.std() * np.sqrt(240))
        }

    return cum_ret_dic, report


#输入参数data为包含因子值得原始数据集，num_group为组数，factor为用于排名的因子名称
def get_group(data, num_group=5, factor='r_singal'):
    ranks = data[factor].rank(ascending=False)  #按降序排名，组号越大，越好
    label = ['G' + str(i) for i in range(1, num_group + 1)]  #创建组号
    category = pd.cut(ranks, bins=num_group, labels=label)
    category.name = 'GROUP'
    new_data = data.join(category)  #将排名合并入原始数据集中
    return new_data

##########################################################
######################敏感性分析用函数####################
# 敏感性分析
def threshold_analysis(df,
                       threshold_col,
                       holding=[5, 10, 15, 20, 25],
                       params='sharpe'):
    ranks = df[threshold_col].rank(ascending=False)
    df['SCORE'] = ranks
    num = 5
    g = pd.cut(ranks.values, 5)
    threshold = g.categories.left.tolist()[1:]
    temp = []
    for i in threshold:
        for j in holding:
            bt = threshold_test(df, 'SCORE', i, j)
            if params == 'ret':
                cum_ret = bt['cum_ret']
                temp.append('{:.2%}'.format(cum_ret[-1]**(240 /
                                                          (len(cum_ret) - 5)) -
                                            1))
            else:
                ret = bt['ret']
                temp.append(ret.mean() / ret.std() * np.sqrt(240))
    temp = np.array(temp).reshape((len(threshold), len(holding)))
    columns_name=list(map(lambda x:'持有{}天'.format(x),holding))
    temp_df = pd.DataFrame(temp, index=threshold, columns=columns_name)
    return temp_df


# 敏感性分析回测用
def threshold_test(df, singal_col, threshold, holding):
    count = holding
    position = []
    for i in range(len(df)):
        factor_value = df[singal_col][i]
        if factor_value > threshold:
            position.append(1)
            count = 1
        else:
            if count < holding:
                count += 1
                position.append(1)
            else:
                position.append(0)
    df['position'] = position
    # 计算收益率
    index_ret = df.close.pct_change().values
    ret = [0]
    # 确定哪些是开仓位置
    for i in range(len(df) - 1):
        ret.append(index_ret[i + 1] * position[i])
    ret = np.array(ret)
    df['ret'] = ret
    cum_ret = []
    # 计算净值
    for i in range(len(ret)):
        if i == 0:
            cum_ret.append(1 + ret[i])
        else:
            cum_ret.append(cum_ret[-1] * (1 + ret[i]))
    df['cum_ret'] = cum_ret
    return df

#############################################################
###############################################################    
# 获取数据
hs300_df = get_datas('000300.XSHG', start, end, interval, 'hs300_datas.pkl')
zz500_df = get_datas('000905.XSHG', start, end, interval, 'zz500_datas.pkl')
cyb_df = get_datas('399006.XSHE', start, end, interval, 'cyb_datas.pkl')
sz50_df = get_datas('000016.XSHG', start, end, interval, 'sz50_datas.pkl')
szzs_df=get_datas('000001.XSHG', start, end, interval, 'szzs_datas.pkl')
zbzs_df=get_datas('399101.XSHE', start, end, interval, 'zbzs_datas.pkl')
 
    