# coding=utf-8
from numpy.core.fromnumeric import product
from numpy.core.numeric import NaN
from jqdatasdk import *
import jqdatasdk as jq
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import *
from pandas.io.pytables import performance_doc
import statsmodels.api as sm
from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
jqdatasdk.auth(13581768635,'Huangtuo05')

#流通市值加权平均
def get_CriValue(exchange_code, start_date, end_date, count=3000):
    '''
    获取流通市值
    参数：
    exchange_code 指数代码，
    start_date 开始日期
    end_date 结束日期
    count 最多返回值个数，是聚宽规定的一次获取最多个数
    返回值：dataframe类型，每日流通市值
    '''
    CriValue = jq.finance.run_query(
            query(finance.STK_EXCHANGE_TRADE_INFO.circulating_market_cap,
                finance.STK_EXCHANGE_TRADE_INFO.date).filter(
                jq.finance.STK_EXCHANGE_TRADE_INFO.exchange_code==exchange_code,
                jq.finance.STK_EXCHANGE_TRADE_INFO.date<=end_date,
                jq.finance.STK_EXCHANGE_TRADE_INFO.date>=start_date).limit(count))
    CriValue.set_index('date', inplace=True)
    return CriValue
# 计算2017年一年的Rm
SH_CriValue = get_CriValue('322001', '2017-01-01', '2017-12-31')
SZ_CriValue = get_CriValue('322004', '2017-01-01', '2017-12-31')
###运用流通市值加权平均法计算Rm###
sz = pd.DataFrame(SZ_CriValue.circulating_market_cap)
sh = pd.DataFrame(SH_CriValue.circulating_market_cap)
real_market = pd.concat([sz,sh],axis=1)
real_market.columns = ['深证','上证']
real_market.index = SH_CriValue.index
real_market['all'] = real_market['深证'] + real_market['上证']
# print(real_market.tail())
SH_market_return = jq.get_price('000001.XSHG',start_date='2017-01-01',end_date='2017-12-31',fields='close'
                             ).pct_change(1).fillna(0)
SZ_market_return = jq.get_price('399106.XSHE',start_date='2017-01-01',end_date='2017-12-31',fields='close'
                             ).pct_change(1).fillna(0)
real_market['R深证'],real_market['R上证'] = SZ_market_return,SH_market_return
Rm = (real_market['深证']/real_market['all'] * real_market['R深证']
                               )+(
      real_market['上证']/real_market['all'] * real_market['R上证'])

#简化的综合市场回报率
def get_Rm_simple_cont(start_date, end_date):
    '''
    功能：计算沪深A股收益率均值作为市场均值
    连续时间
    参数：start_date, end_date
    返回值：list类型，A股收益率
    '''
    SH_market_return = jq.get_price('000002.XSHG',start_date=start_date,end_date=end_date,fields=['close', 'pre_close']
                                ).fillna(0)
    SH_market_return = (SH_market_return['close']-SH_market_return['pre_close']) / SH_market_return['pre_close']
    SZ_market_return = jq.get_price('399107.XSHE',start_date=start_date,end_date=end_date,fields=['close', 'pre_close']
                                ).fillna(0)
    SZ_market_return = (SZ_market_return['close']-SZ_market_return['pre_close']) / SZ_market_return['pre_close']
    Rm_simple_cont = (SH_market_return+SZ_market_return)/2
    return list(Rm_simple_cont)

#计算其他指数的收益率
def get_Rm_simple2(start_date,end_date,time_interval=None,code = None):
    '''
    功能：计算沪深A股收益率均值作为市场均值，也可以计算其他指数的收益率
    参数：start_date,end_date,
        time_interval默认为None,计算每日收益率,当为int型数字时，
        计算前time_interval天的指数平均收益率
    返回值：list类型，A股收益率
    '''
    if not code:
        return0 = jq.get_price(['000002.XSHG','399107.XSHE'],start_date=start_date,end_date=end_date,fields=['close', 'pre_close'],
        ).fillna(0)
        SH_market_return = return0.loc[return0['code']=='000002.XSHG'].reset_index(drop=True)
        SZ_market_return = return0.loc[return0['code']=='399107.XSHE'].reset_index(drop=True)
if time_interval:
    return1 = jq.get_price(['000002.XSHG','399107.XSHE'],count = time_interval,end_date=start_date,fields=['close', 'pre_close'],
    ).fillna(0)
    SH_market_return1 = return1.loc[return1['code']=='000002.XSHG'].reset_index(drop=True)
    SH_market_return1.drop([len(SH_market_return1)-1],inplace=True)
    SH_market_return = SH_market_return1.append(SH_market_return).reset_index(drop = True)
    SZ_market_return1 = return1.loc[return1['code']=='399107.XSHE'].reset_index(drop=True)
    SZ_market_return1.drop([len(SZ_market_return1)-1],inplace=True)
    SZ_market_return = SZ_market_return1.append(SZ_market_return).reset_index(drop = True)
    SH_market_return = (SH_market_return['close']-SH_market_return['pre_close']) / SH_market_return['pre_close']
    SZ_market_return = (SZ_market_return['close']-SZ_market_return['pre_close']) / SZ_market_return['pre_close']
    Rm_simple = list((SH_market_return+SZ_market_return)/2)
else:
    return0 = jq.get_price(code,start_date=start_date,end_date=end_date,fields=['close', 'pre_close'],
    ).fillna(0).reset_index(drop=True)
if time_interval:
    return1 = jq.get_price(code,count = time_interval,end_date=start_date,fields=['close', 'pre_close'],
    ).fillna(0)
    return1 = return1.reset_index(drop=True)
    return1.drop([len(return1)-1],inplace=True)
    return0 = return1.append(return0).reset_index(drop = True)
    return0 = (return0['close']-return0['pre_close']) / return0['pre_close']
    Rm_simple = list(return0)
if time_interval:
    Rm_simple.append(0)
    Rm_simple = [(np.prod(1+np.array(Rm_simple[i-time_interval:i])))**(1/time_interval)-1 for i in range(time_interval, len(Rm_simple))]
return Rm_simple

def get_stocks(date,index=None):
    '''
    功能：
    根据日期，获取该日满足交易要求的股票相关数据，即剔除ST股、上市未满60天、停牌、跌涨停股
    参数：
    date，日期
    index，指数代码，在特定指数的成分股中选股。缺省时选股空间为全部A股
    返回：
    DataFrame类型，索引为股票代码，同时包含了价格数据，方便后续使用
    '''
stocks = jq.get_all_securities(
types=['stock'],
date=date
)#该日正在上市的股票

if index:#特定成分股
    stock_codes = jq.get_index_stocks(index,date=date)#成分股
    stocks = stocks[stocks.index.isin(stock_codes)]

#上市日期大于60个自然日
# start_date 为 [datetime.date] 类型
stocks['start_date']=pd.to_datetime(stocks['start_date'])
date = datetime.strptime(date,'%Y-%m-%d').date()
date = pd.to_datetime(date)
stocks['datedelta'] = date - stocks['start_date']
stocks = stocks[stocks['datedelta'] > timedelta(days=60)]

#是否是ST股
stocks['is_st'] = jq.get_extras(
info='is_st',
security_list=list(stocks.index),
count=1,
end_date=date
).T

#涨停、跌停、停牌
stocks_info = jq.get_price(
security = list(stocks.index),
fields=['close','high','low','high_limit','low_limit','paused'],
count=1,
end_date=date,
panel=False
).set_index('code').drop('time',axis=1)

stocks['price'] = stocks_info['close']#顺便保存价格，方便后续运算
stocks['paused'] = stocks_info['paused'] == 1#是否停牌
stocks['high_stop'] = stocks_info['high'] >= stocks_info['high_limit']#涨停
stocks['low_stop'] = stocks_info['low'] <= stocks_info['low_limit']#跌停
stocks = stocks[~(stocks['is_st'] | stocks['paused'] | stocks['high_stop'] | stocks['low_stop'])]
return stocks

# ts日期转聚宽日期
def ts2jq_date(tsdate):
    '''
    tsdate必须是list
    输出list
    '''
date=['-'.join([i[0:4], i[4:6], i[6:]]) for i in tsdate]
return date
# 聚宽日期转ts日期只需要date.replace('-', '')删除'-'即可
# 聚宽股票代码转ts股票代码
def jq2ts_code(jqcode):
    '''
    输入聚宽的股票代码(list)
    返回tushare股票代码(list)
    '''
tscode = [i.replace('XSHE', 'SZ').replace('XSHG', 'SH') for i in jqcode]
return tscode

def cal_CSAD_cont(start_date, end_date, date_interval, index):
    '''
    为了减少运行时间，股票不能每日筛选
    date_interval:股票池更新间隔,int
    '''
CSAD = pd.DataFrame([])
start_date_ts = start_date.replace('-', '')
end_date_ts = end_date.replace('-', '')
trade_days = pro.trade_cal(start_date=start_date_ts, end_date=end_date_ts)
trade_days = trade_days[trade_days['is_open']==1]['cal_date']
trade_days = list(trade_days)
trade_days = ts2jq_date(trade_days)
dates_update = trade_days[::date_interval]
CSAD['cal_date'] = trade_days
csad = []
stocks_rm = get_Rm_simple2(start_date, end_date)
for i in range(len(trade_days)):
    if trade_days[i] in dates_update:  #如果在dates_update就更新stocks
        stocks = get_stocks(trade_days[i], index)
        stocks = list(stocks.index)
        stocks = jq2ts_code(stocks)
        ts_date = trade_days[i].replace('-', '')
        ri = pro.daily(trade_date=ts_date, fields='ts_code,pct_chg')
        ri = ri[ri.ts_code.isin(stocks)].drop('ts_code',axis=1)
        csadi = float(np.mean(abs(ri-stocks_rm[i])))
        csad.append(csadi)
CSAD['CSAD'] = csad
CSAD['rm'] = stocks_rm
CSAD.set_index('cal_date')
return CSAD

#对CSAD和综合市场收益率Rm进行OLS拟合，计算拟合的二次项系数beta2以及p值
def cal_beta(start_date, end_date, date_interval, index):
# 返回start_date22天后的beta
# 想要返回从start_date后的beta请使用
# real_date = get_trade_day_before(22,start_date)
# start_date = real_date
CSAD = cal_CSAD_cont(start_date, end_date, date_interval, index)
x1=[]
x2=[]
p1 = []
p2 = []
rm_mean = []
for i in range(len(CSAD))[22:]:
    x=CSAD.iloc[i-22:i, 2] #pd.ix方法被1.0.0版本移除了
    y=CSAD.iloc[i-22:i, 1]
    X = np.column_stack((x**2, x))
    X = sm.add_constant(X)
    model = sm.OLS(y,X)
results = model.fit()
x1.append(results.params[1])
x2.append(results.params[2])
p1.append(results.pvalues[1])
p2.append(results.pvalues[2])
meanrm = np.mean(x)
rm_mean.append(meanrm)
beta = pd.DataFrame([])
beta['beta1']=x1
beta['pvalue1']=p1
beta['beta2']=x2
beta['pvalue2']=p2
beta['date']=list(CSAD.cal_date[22:])
beta['rm']=rm_mean
beta.set_index('date',inplace = True)#默认的drop=True失效了是怎么回事
return beta

#以沪深300为例
HS300_beta = cal_beta('2007-01-01','2019-01-01',180, '000300.XSHG')
HS300_beta.to_excel(excel_writer='beta.xlsx', sheet_name='sheet_1')
# 绘制beta2的p值
df1 = pd.read_excel('beta.xlsx')
print(df1)
fig,ax=plt.subplots(figsize=(13,8))
plt.plot(df1.date,df1.pvalue2,'b',linewidth = 1)
plt.grid(False)
plt.title('07年1月1日至19年1月β2的p值折线走势图',fontsize=20)
plt.hlines(y=0.05,xmin=df1.date[0],xmax=df1.date[len(df1)-1],color='red',linewidth = 1,alpha = 0.5)
first_legend = plt.legend(['pvlaue-β2'],fontsize=15)
ax1 = plt.gca().add_artist(first_legend)
red_patch = mpatches.Patch(color = 'red',label = '0.05分界线',linewidth = 1,alpha = 0.5)
plt.legend(handles = [red_patch],loc = 3,fontsize=15)
plt.xlabel('时间',fontsize=15)
plt.ylabel('pvalue',fontsize=15)
for label in ax.xaxis.get_ticklabels():
# label is a Text instance
label.set_color('red')
label.set_rotation(45)
label.set_fontsize(10)
for label in ax.get_xticklabels():
label.set_visible(False)
for label in ax.get_xticklabels()[::180]:
label.set_visible(True)
plt.tight_layout()
ax.tick_params(bottom=False,top=False,left=False,right=False)  #移除全部刻度线
plt.show()
fig,ax=plt.subplots(figsize=(13,8))
plt.plot(df1.date,df1.beta2,'r')
plt.grid(False)
plt.title('07年1月1日至19年1月β2的折线走势图',fontsize=20)
plt.legend(['β2'],fontsize=15)
plt.xlabel('时间',fontsize=15)
plt.ylabel('β2数值',fontsize=15)
for label in ax.xaxis.get_ticklabels():
# label is a Text instance
label.set_color('red')
label.set_rotation(45)
label.set_fontsize(10)
for label in ax.get_xticklabels():
label.set_visible(False)
for label in ax.get_xticklabels()[::180]: #每180个刻度显示一个
label.set_visible(True)
plt.tight_layout()
ax.tick_params(bottom=False,top=False,left=False,right=False)  #移除全部刻度线
plt.show()

#方法一
#通常的ols拟合，当p值小于0.05时，认为beta2显著。但是可以看出该方法给出的开仓信号过多。一个方法是将临界的p值减小，例如减小到0.001
# 筛选出显著羊群效应即β^2显著的，p值<0.001
df1 = pd.read_excel('beta.xlsx')
pvalues = df1.pvalue2
rm22_mean = df1.rm
signal_up = [rm22_mean[i]>0 and pvalues[i]<0.001 for i in range(len(pvalues))]
signal_down  = [rm22_mean[i]<0 and pvalues[i]<0.025 for i in range(len(pvalues))]
print('上涨时置信水平超过99.9%的信号出现了{}次，\n\
下跌时置信水平超过97.25%的信号出现了{}次，\n\
共出现信号{}次\n'.format(sum(signal_up),sum(signal_down),sum(signal_up)+sum(signal_down)))

# 可视化
start_date = df1.date[0]
end_date = list(df1['date'])[-1]
###信号坐标##
R_300 = jq.get_price('000300.XSHG',start_date=start_date,end_date=end_date,fields=['low','close'])
fig = plt.subplots(figsize=(13,8))
plt.plot(R_300.close,'b')
plt.grid(True)
plt.title('单侧置信区间下限0.1%时的β信号分布图',fontsize=20)
Y = R_300.close
for i in range(len(signal_up)):
if signal_up[i]:
loc = int(Y[i])
plt.vlines(Y.index[i], loc-400,loc+400, color='red',linewidth = 1,alpha = 0.5)
first_legend = plt.legend(['000300.XSHG'],fontsize=20)
ax = plt.gca().add_artist(first_legend)
blue_patch = mpatches.Patch(color = 'red',label = '买入信号',linewidth = 1,alpha = 0.5)
plt.legend(handles = [blue_patch],loc = 4,fontsize=20)
plt.xlabel('时间',fontsize=15)
plt.ylabel('沪深300指数收盘价',fontsize=15)
plt.show()

#方法二
#使用某日前180日的所有ols拟合的β2作为总体的β2分布，用其上、下0.05分位数作为总体的置信限。
def get_trade_day_before(count, end_date):
'''
获取end_date前count天的交易日，从该日到end_date一共count天
'''
return1 = jq.get_price('000002.XSHG',count = count,end_date=end_date,fields='close').fillna(0)
trade_day_before = return1.index[0]
return str(trade_day_before)[:10]

# 增加一个均线策略
######################################################################
def get_ma(start_date,end_date,time_interval,code = None):
'''
功能：计算沪深A股time_interval天均值作为市场均值
参数：start_date,end_date,
time_interval为int型数字
code为可选参数，默认计算沪深A股均值。也可以选择其他指数
计算前time_interval天的平均值
返回值：list类型，time_interval天的平均值
'''
if not code:
# 将time_interval天的指数价格拼接在start_date前面，
# 然后rolling方法计算均值，最后丢掉前面的指数价格
return0 = jq.get_price(['000002.XSHG','399107.XSHE'],start_date=start_date,end_date=end_date,fields=['close','pre_close']
).fillna(0)
SH_market_return = return0.loc[return0['code']=='000002.XSHG'].reset_index(drop=True)
SZ_market_return = return0.loc[return0['code']=='399107.XSHE'].reset_index(drop=True)
return1 = jq.get_price(['000002.XSHG','399107.XSHE'],count = time_interval,end_date=start_date,fields=['close','pre_close']
).fillna(0)
SH_market_return1 = return1.loc[return1['code']=='000002.XSHG'].reset_index(drop=True)
SH_market_return1.drop([len(SH_market_return1)-1],inplace=True)
SH_market_return = SH_market_return1.append(SH_market_return).reset_index(drop = True)
SZ_market_return1 = return1.loc[return1['code']=='399107.XSHE'].reset_index(drop=True)
SZ_market_return1.drop([len(SZ_market_return1)-1],inplace=True)
SZ_market_return = SZ_market_return1.append(SZ_market_return).reset_index(drop = True)
SH_market_return = SH_market_return.close
SZ_market_return = SZ_market_return.close
ma_simple = (SH_market_return+SZ_market_return)/2
else:
return0 = jq.get_price(code,start_date=start_date,end_date=end_date,fields=['close','pre_close'],
).fillna(0)
return1 = jq.get_price(code,count = time_interval,end_date=start_date,fields=['close', 'pre_close'],
).fillna(0)
return1 = return1.reset_index(drop=True)
return1.drop([len(return1)-1],inplace=True)
return0 = return1.append(return0).reset_index(drop = True)
return0 = return0.close
ma_simple = return0
ma = ma_simple.rolling(time_interval,min_periods=time_interval).mean()
ma.drop(ma.index[:time_interval-1], axis=0, inplace = True)
return ma

#分析ols拟合后的数据，计算beta,MA5,MA10,MA20,22天平均收益率,180天2.5%,5%,10%分位数。
# 获取标的指数的beta,MA5,MA10,MA20,22天平均收益率,180天2.5%,5%,10%的分位数
# 输入的数据为cal_beta()的返回dataframe以及指数代码
def get_analysis_frame(code, beta_frame):
'''
标的指数的beta,MA5,MA10,MA20,22天平均收益率,180天2.5%,5%,10%的分位数
'''
# 直接从beta_frame中读取日期
start_date = beta_frame.iloc[0,0]
end_date = beta_frame.iloc[len(beta_frame)-1,0]
beta = beta_frame[['date','beta2','pvalue2']]
R_MA22 = get_Rm_simple2(start_date, end_date, 22, code)
MA30 = get_ma(start_date, end_date, 30, code)
# 获取30个交易日前的日期
start_date_before = get_trade_day_before(30, start_date)
MA30_before = get_ma(start_date_before, end_date, 30, code)
# 30个交易日前的日期到结束日期前30日的均线
MA30_before = MA30_before[:-29]
MA5_before = get_ma(start_date_before, end_date, 5, code)
MA5_before = MA30_before[:-4]
MA10 = get_ma(start_date, end_date, 10, code)
MA5 = get_ma(start_date, end_date, 5, code)
beta['MA5'] = MA5
beta['MA5_before'] = MA5_before
beta['MA10']= MA10
beta['MA30']= MA30
beta['MA30_before'] = MA30_before
beta['R_MA22'] = R_MA22
beta['quantile-0.1'] = beta['beta2'].rolling(180,min_periods=180).quantile(0.1)
beta['quantile-0.05'] = beta['beta2'].rolling(180,min_periods=180).quantile(0.05)
beta['quantile-0.025'] = beta['beta2'].rolling(180,min_periods=180).quantile(0.025)

"""为了避免过拟合的情况，分位数的计算方式为向前滚动计算180天的分位数
180天即半年，是许多标的指数成份股的调整周期
"""

beta = beta.fillna(method = 'bfill')
return beta

#有均线时：
df1 = pd.read_excel('beta.xlsx')
analysis = get_analysis_frame('000300.XSHG',df1)
ma30 = analysis.MA30
ma5 = analysis.MA5
ma30_before = analysis.MA30_before
ma5_before = analysis.MA5_before
rm22_mean = analysis.R_MA22
beta2 = analysis.beta2
q_0_05 = analysis['quantile-0.05']
q_01 = analysis['quantile-0.1']
# 增加条件：三十日均线大于三十日前的均线
# 线性平均和指数平均我觉得没啥大的区别
# MA30如果下降，rm30也应该下降
signal_up = [rm22_mean[i]>0 and beta2[i]<q_01[i] and ma30[i] > ma30_before[i] for i in range(len(beta2))]
signal_down  = [rm22_mean[i]<0 and beta2[i]<q_01[i] and ma5[i] < ma5_before[i] for i in range(len(beta2))]
df1['signal_up'] = signal_up
df1['signal_down'] = signal_down
print('上涨时超过单侧置信区间百分之5下限的beta信号出现了{}次，\n\
下跌时超过单侧置信区间百分之5下限的beta信号出现了{}次，\n\
共出现单侧置信区间下限为百分之10的beta信号{}次\n'.format(sum(signal_up),sum(signal_down),sum(signal_up)+sum(signal_down)))
start_date = df1.date[0]
end_date = list(df1['date'])[-1]
###信号坐标##
R_300 = jq.get_price('000300.XSHG',start_date=start_date,end_date=end_date,fields=['low','close'])
fig = plt.subplots(figsize=(13,8))
plt.plot(R_300.close,'b')
plt.grid(True)
plt.title('图七 改进后的多空信号图',fontsize=20)
Y = R_300.close
for i in range(len(signal_up)):
if signal_up[i]:
loc = int(Y[i])
plt.vlines(Y.index[i], loc-400,loc+400, color='red',linewidth = 1,alpha = 0.5)
if signal_down[i]:
loc = int(Y[i])
plt.vlines(Y.index[i], loc-400,loc+400, color='green',linewidth = 1,alpha = 0.5)
first_legend = plt.legend(['000300.XSHG'],fontsize=20)
ax = plt.gca().add_artist(first_legend)
blue_patch = mpatches.Patch(color = 'red',label = '做多信号',linewidth = 1,alpha = 0.5)
green_patch = mpatches.Patch(color = 'green',label = '做空信号',linewidth = 1,alpha = 0.5)
plt.legend(handles = [blue_patch,green_patch],loc = 4,fontsize=20)
plt.xlabel('时间',fontsize=15)
plt.ylabel('沪深300指数收盘价',fontsize=15)
plt.show()

#3.4策略构建和回测
#宽基指数
#定义一个计算所有感兴趣信息的函数
def cal_rate(code, start_date, end_date):
    '''
    获取信号发生后的收益率，胜率，夏普比率等若干信息。分上涨和下跌两种
    参数：标的指数代码，回测开始和结束日期
    返回值：
    回测的数据dataframe类型，包含多空次数，胜率，最大回撤，夏普比率
    累计策略收益率，dataframe类型，分做空和做多
    每次做多/空的收益率
    '''
    try:
        df1 = pd.read_excel('{}_{}_{}.xlsx'.format(code,start_date,end_date))
    except:
        real_date = get_trade_day_before(22,start_date)
        df1 = cal_beta(real_date,end_date,180,code)
        df1.to_excel(excel_writer='{}_{}_{}.xlsx'.format(code,start_date,end_date), sheet_name='sheet_1')
        df1 = pd.read_excel('{}_{}_{}.xlsx'.format(code,start_date,end_date))
    analysis = get_analysis_frame(code,df1)
    ma30 = analysis.MA30
    ma5 = analysis.MA5
    ma30_before = analysis.MA30_before
    ma5_before = analysis.MA5_before
    rm22_mean = analysis.R_MA22
    beta2 = analysis.beta2
    q_0_05 = analysis['quantile-0.05']
    q_01 = analysis['quantile-0.1']
    # 增加条件：三十日均线大于三十日前的均线
    # 线性平均和指数平均我觉得没啥大的区别
    # MA30如果下降，rm30也应该下降
    signal_up = [rm22_mean[i]>0 and beta2[i]<q_01[i] and ma30[i] > ma30_before[i] for i in range(len(beta2))]
    signal_down  = [rm22_mean[i]<0 and beta2[i]<q_01[i] and ma5[i] < ma5_before[i] for i in range(len(beta2))]
    # 次日持仓情况
    # 是否开仓标志
    is_hold_open = [False]
    is_sell_open = [False]
    for i in range(1,len(signal_up)):
        if signal_up[i-1] and not any(signal_up[max(i-23,0):i-1]):
            is_hold_open.append(True)
        else:
            is_hold_open.append(False)
        if signal_down[i] and not any(signal_down[max(i-22,0):i]):
            is_sell_open.append(True)
        else:
            is_sell_open.append(False)
    df1['is_hold_open'] = is_hold_open
    df1['is_sell_open'] = is_sell_open
    # 次日是否持仓标记
    is_hold = [any(is_hold_open[max(0,i-22):i]) for i in range(len(is_hold_open))]
    is_sell = [any(is_sell_open[max(0,i-22):i]) for i in range(len(is_sell_open))]
    # 指数日收益率
    return0 = jq.get_price(code,start_date=start_date,end_date=end_date,fields=['close', 'pre_close'],
                                ).fillna(0)
    return0['rates'] = return0['close'] / return0['pre_close'] - 1
    # 当日收益率，空仓时按照1计算
    rates_up = [return0.iloc[i,2]+1 if is_hold[max(i-1,0)] else 1 for i in range(len(return0))]
    rates_down = [-return0.iloc[i,2]+1 if is_sell[max(i-1,0)] else 1 for i in range(len(return0))]
    rates_up = np.array(rates_up)
    rates_down = np.array(rates_down)
    df1['rates_up'] = rates_up
    df1['rates_down'] = rates_down
    # 开仓次数
    open_up_times = sum(is_hold_open)
    open_down_times = sum(is_sell_open)
    rates_up_cum = []
    rates_down_cum = []
    # 分日累计收益率
    rates_daily_up = []
    rates_daily_down = []
    for i in range(len(is_hold_open)):
        if is_hold_open[i]:
            rates_up_cum.append(product(rates_up[i+1:i+23]))
            rates_daily_up.append(rates_up[i+1:i+23])
        if is_sell_open[i]:
            rates_down_cum.append(product(rates_down[i+1:i+23]))
            rates_daily_down.append(rates_down[i+1:i+23])
    rates_up_cum=np.array(rates_up_cum)
    rates_down_cum=np.array(rates_down_cum)
    # 胜次数
    up_win_times = sum(rates_up_cum>1)
    down_win_times = sum(rates_down_cum>1)
    # 胜率
    if open_up_times:
        win_rate_up = up_win_times/open_up_times
    else:
        win_rate_up = NaN
    if open_down_times:
        win_rate_down = down_win_times/open_down_times
    else:
        win_rate_down = NaN
    ###计算最大回撤，只计算做多###
    re = []
    for k in range(len(rates_up)):
        retreat = max(rates_up[k]-rates_up[k:])
        re.append(retreat)
    max_retreat = max(re)
    ###计算夏普比率，只计算做多###
    ex_pct_close = rates_up - 1 - 0.04/252
    sharpe = (ex_pct_close.mean() * (252)**0.5)/ex_pct_close.std()
    analysis_datas = np.array([open_up_times,open_down_times,win_rate_up,win_rate_down,max_retreat,sharpe]).reshape(1,6)
    back_analysis_data = pd.DataFrame(analysis_datas,
                                        columns =['做多次数','做空次数','做多胜率','做空胜率','最大回撤','夏普比率'
                                        ],index = [code])
    rates_analysis = df1[['date','rates_up','rates_down']]
    rates_up_prod = [product(rates_up[:i]) for i in range(len(rates_up))]
    rates_down_prod = [product(rates_down[:i]) for i in range(len(rates_down))]
    rates_analysis['rates_up_prod'] = rates_up_prod
    rates_analysis['rates_down_prod'] = rates_down_prod
    return back_analysis_data,rates_analysis,rates_daily_up,rates_daily_down,rates_up_cum,rates_down_cum

#配合下面的函数使用，获得分年度的策略表现。
def get_annual_performance(time_interval,code):
    # 对于某一指数的若干年的数据
    # 返回一个dataframe
    # 包含每年做多次数，胜率，做空次数，胜率，最大回撤，夏普比率
    performance = pd.DataFrame(columns =['做多次数','做多胜率','最大回撤-多','夏普比率-多','做空次数','做空胜率','最大回撤-空','夏普比率-空'])
    for i in range(len(time_interval)-1):
        start_date = time_interval[i]
        end_date = time_interval[i+1]
        back_analysis_data=cal_rate(code, start_date, end_date)[0].reset_index(drop=True)
        performance = performance.append(back_analysis_data, ignore_index=True)
    time_index = [i[0:4] for i in time_interval[:-1]]
    performance['年份'] = time_index
    performance.set_index('年份',inplace=True)
    return performance

#展示沪深300指数14-15年两年
def get_everyyear_rate(time_interval, code):
    # 对于某一指数的若干年的数据
    # 返回一个dataframe
    # 包含当日收益率和从开始到结束日期的累计收益率，分为做多和做空
    rate = pd.DataFrame(columns=['date', 'rates_up', 'rates_down', 'rates_up_prod', 'rates_down_prod'])
    for i in range(len(time_interval) - 1):
        start_date = time_interval[i]
        end_date = time_interval[i + 1]
        back_analysis_data = cal_rate(code, start_date, end_date)[1]
        rate = rate.append(back_analysis_data, ignore_index=True)

    rate_cum_up = [product(rate.iloc[:i, 1]) for i in range(len(rate))]
    rate['rate_prod_up'] = rate_cum_up
    rate_cum_down = [product(rate.iloc[:i, 2]) for i in range(len(rate))]
    rate['rate_prod_down'] = rate_cum_down
    rate.set_index('date', inplace=True)
    return rate

#研报中提到做空的羊群效应持续时间很短，可以考虑对做空时间缩短
code = '000300.XSHG'# 上证300
start_date = '2014-01-01'
end_date = '2016-01-01'

back_analysis_data,rates_analysis,rates_daily_up,rates_daily_down,rates_up_cum,rates_down_cum = cal_rate(code, start_date, end_date)
print(back_analysis_data)
rates_daily_down0 = rates_daily_down[0]
# print(rates_daily_down0)
rates_daily_down_cum = [product(rates_daily_down0[:i]) for i in range(len(rates_daily_down0))]
# print(rates_daily_down_cum)
fig = plt.subplots(figsize=(13,8))
plt.plot(rates_daily_down_cum,'green')
plt.grid(True)
plt.title('图八，做空开仓后22日的累计收益率',fontsize=20)
plt.xlabel('时间',fontsize=15)
plt.ylabel('沪深300某次做空的收益率',fontsize=15)
plt.show()

rates_daily_up0 = rates_daily_up[0]
# print(rates_daily_down0)
rates_daily_up_cum = [product(rates_daily_up0[:i]) for i in range(len(rates_daily_up0))]
fig = plt.subplots(figsize=(13,8))
plt.plot(rates_daily_up_cum,'green')
plt.grid(True)
plt.title('图八，做多开仓后22日的累计收益率',fontsize=20)
plt.xlabel('时间',fontsize=15)
plt.ylabel('沪深300某次做多的收益率',fontsize=15)
plt.show()

#下面以上证50为例，对收益率可视化
'''
上证50
'''
code = '000016.XSHG'# 上证50
# 时间取07-17年
time_interval = ['2007-01-01','2008-01-01','2009-01-01']
time_interval.extend(['20{}-01-01'.format(i) for i in range(10,17+2)])
performance = get_annual_performance(time_interval,code)
performance.to_excel(excel_writer='07-17年上证50表现.xlsx', sheet_name='sheet_1')
start_date = '2007-01-01'
end_date = '2008-01-01'
back_analysis_data,rates_analysis,rates_daily_up,rates_daily_down,rates_buy_each_time,rates_sell_each_time = cal_rate(code, start_date, end_date)

rates_daily_up0 = rates_daily_up[0]
# print(rates_daily_down0)
rates_daily_up_cum = [product(rates_daily_up0[:i]) for i in range(len(rates_daily_up0))]
fig = plt.subplots(figsize=(13,8))
plt.plot(rates_daily_up_cum,'green')
plt.grid(True)
plt.title('图八，做多开仓后22日的累计收益率',fontsize=20)
plt.xlabel('时间',fontsize=15)
plt.ylabel('07-08年上证50某次做多的收益率',fontsize=15)
plt.show()

fig,ax=plt.subplots(figsize=(13,8))
plt.plot(rates_analysis.date,rates_analysis.rates_up_prod,'b')
plt.grid(False)
plt.title('图九，07-08年上证50做多收益率',fontsize=20)
plt.legend(['收益率'],fontsize=15)
plt.xlabel('时间',fontsize=15)
plt.ylabel('收益率',fontsize=15)
for label in ax.xaxis.get_ticklabels():
    # label is a Text instance
    label.set_color('red')
    label.set_rotation(45)
    label.set_fontsize(10)
for i,label in enumerate(ax.get_xticklabels()):
    if i%30==0:
        label.set_visible(True)
    else:
        label.set_visible(False)

plt.tight_layout()
ax.tick_params(bottom=False,top=False,left=False,right=False)  #移除全部刻度线
plt.show()

#最后，实现对07-08年上证综指、上证50、沪深300、中小板综合指数的回测。
'''
宽基指数做多收益率
'''
# 不知道什么原因，中证500和创业板指获取不了成分股
index_code = ['000001.XSHG','000016.XSHG','000300.XSHG'
              ,'399101.XSHE']
colors = ['red','orange','yellow','green']
# 时间取07-17年
time_interval = ['2007-01-01','2008-01-01','2009-01-01']
# time_interval.extend(['20{}-01-01'.format(i) for i in range(10,17+2)])

fig,ax=plt.subplots(figsize=(13,8))
for i,code in enumerate(index_code):
    every_rate = get_everyyear_rate(time_interval, code)
    plt.plot(every_rate.index,every_rate.rate_prod_up,color = colors[i],label = code)
plt.grid(False)
plt.title('07-08年宽基指数做多收益率',fontsize=20)
plt.legend()
plt.xlabel('时间',fontsize=15)
plt.ylabel('收益率',fontsize=15)
for label in ax.xaxis.get_ticklabels():
    # label is a Text instance
    label.set_color('red')
    label.set_rotation(45)
    label.set_fontsize(10)
for i,label in enumerate(ax.get_xticklabels()):
    if i%90==0:
        label.set_visible(True)
    else:
        label.set_visible(False)

plt.tight_layout()
ax.tick_params(bottom=False,top=False,left=False,right=False)  #移除全部刻度线
plt.show()


'''
上涨时分日累计收益率
'''
index_code = ['000001.XSHG','000016.XSHG','000300.XSHG'
              ,'399101.XSHE']
colors = ['red','orange','yellow','green']
# 时间取07-17年
time_interval = ['2007-01-01','2008-01-01','2009-01-01']
# time_interval.extend(['20{}-01-01'.format(i) for i in range(10,17+2)])

fig,ax=plt.subplots(figsize=(13,8))

for i,code in enumerate(index_code):
    rates_daily_up = []
    for j in range(len(time_interval)-1):
        rates_daily_up.extend(cal_rate(code, time_interval[j], time_interval[j+1])[2])
    rates_daily_up0 = rates_daily_up[1]
    rates_daily_up_cum = [product(rates_daily_up0[:i]) for i in range(len(rates_daily_up0))]
    plt.plot(rates_daily_up_cum,color = colors[i],label = code)
plt.grid(False)
plt.title('07-08年上涨时分日累计收益率',fontsize=20)
plt.legend()
plt.xlabel('时间',fontsize=15)
plt.ylabel('收益率',fontsize=15)

plt.tight_layout()
plt.show()


'''
下跌时分日累计收益率
'''
index_code = ['000001.XSHG','000016.XSHG','000300.XSHG'
              ,'399101.XSHE']
colors = ['red','orange','yellow','green']
# 时间取07-17年
time_interval = ['2007-01-01','2008-01-01','2009-01-01']
# time_interval.extend(['20{}-01-01'.format(i) for i in range(10,17+2)])

fig,ax=plt.subplots(figsize=(13,8))

for i,code in enumerate(index_code):
    rates_daily_up = []
    for j in range(len(time_interval)-1):
        rates_daily_up.extend(cal_rate(code, time_interval[j], time_interval[j+1])[3])
    rates_daily_up0 = rates_daily_up[1]
    rates_daily_up_cum = [product(rates_daily_up0[:i]) for i in range(18)]
    plt.plot(rates_daily_up_cum,color = colors[i],label = code)
plt.grid(False)
plt.title('07-08年下跌时分日累计收益率',fontsize=20)
plt.legend()
plt.xlabel('时间',fontsize=15)
plt.ylabel('收益率',fontsize=15)

plt.tight_layout()
plt.show()




