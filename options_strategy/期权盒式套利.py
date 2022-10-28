from jqdata import *
from jqdata import jy
import numpy as np
import pandas as pd
import datetime
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


class option(object):
    def __init__(self, S0, K, T, r, sigma, otype):
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.otype = otype

    def d1(self):
        d1 = ((log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        return d1

    def d2(self):
        d2 = ((log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        return d2

    def value(self):
        d1 = self.d1()
        d2 = self.d2()
        if self.otype == 'Call':
            value = self.S0 * stats.norm.cdf(d1, 0.0, 1.0) - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2, 0.0,
                                                                                                             1.0)
        elif self.otype == 'Put':
            value = self.K * exp(-self.r * self.T) * stats.norm.cdf(-d2, 0.0, 1.0) - self.S0 * stats.norm.cdf(-d1, 0.0,
                                                                                                              1.0)
        else:
            print('option type mistake')
        return value

    def impvol(self, Price):
        a = 0
        b = 1
        while b - a > 0.001:
            self.sigma = (a + b) / 2
            if self.value() >= Price:
                b = self.sigma
            else:
                a = self.sigma
        iv = (a + b) / 2
        return iv


def get_50ETF_option(date, dfSignal=False):
    S0 = get_price('510050.XSHG', start_date=date, end_date=date, fields=['close']).values[0][0]
    r = 0.03

    q = query(jy.Opt_DailyPreOpen).filter(jy.Opt_DailyPreOpen.TradingDate == date,
                                          jy.Opt_DailyPreOpen.ULAName == '50ETF')
    df = jy.run_query(q).loc[:, ['ContractCode', 'TradingCode', 'StrikePrice', 'ExerciseDate']]
    exercise_date_list = sorted(df['ExerciseDate'].unique())

    key_list = []
    Contract_dict = {}
    Price_dict = {}
    impVol_dict = {}

    for exercise_date in exercise_date_list:

        # 获得T型代码
        df1 = df[df['ExerciseDate'] == exercise_date]
        # 去除调整合约
        check = []
        for i in df1['TradingCode']:
            x = True if i[11] == 'M' and i[6] == 'C' else False
            check.append(x)
        df_C = df1[check][['ContractCode', 'StrikePrice']]
        df_C.index = df_C.StrikePrice.values
        del df_C['StrikePrice']
        df_C.columns = ['Call']
        df_C = df_C.sort_index()

        # 去除调整合约
        check = []
        for i in df1['TradingCode']:
            x = True if i[11] == 'M' and i[6] == 'P' else False
            check.append(x)
        df_P = df1[check][['ContractCode', 'StrikePrice']]
        df_P.index = df_P.StrikePrice.values
        del df_P['StrikePrice']
        df_P.columns = ['Put']
        df_P = df_P.sort_index()

        dfT = pd.concat([df_C, df_P], axis=1)
        exercise_date = datetime.datetime.strptime(str(exercise_date)[:10], '%Y-%m-%d')
        exercise_date = datetime.date(exercise_date.year, exercise_date.month, exercise_date.day)

        Contract_dict[exercise_date] = dfT

        # T型价格
        q = query(jy.Opt_DailyQuote).filter(jy.Opt_DailyQuote.TradingDate == date)
        df2 = jy.run_query(q).loc[:, ['ContractCode', 'ClosePrice']]
        df2.index = df2['ContractCode'].values
        del df2['ContractCode']
        dfPrice = dfT.copy()
        dfPrice['Call'] = df2.loc[dfT.loc[:, 'Call'].values, :].values
        dfPrice['Put'] = df2.loc[dfT.loc[:, 'Put'].values, :].values
        dfPrice = dfPrice

        Price_dict[exercise_date] = dfPrice

        dfimpVol = dfPrice.copy()
        T = (exercise_date - date).days / 365
        for K in dfimpVol.index:
            for otype in dfimpVol.columns:
                optionprice = dfPrice.loc[K, otype] + 1.3 / 10000
                x = option(S0, K, T, r, 0, otype)
                dfimpVol.loc[K, otype] = x.impvol(optionprice)

        impVol_dict[exercise_date] = dfimpVol
        key_list.append(exercise_date)

    if dfSignal:
        value_list = []
        for key, value in Contract_dict.items():
            value['exercise_date'] = key
            value_list.append(value)
        Contract_df = pd.concat(value_list).sort('exercise_date')

        value_list = []
        for key, value in Price_dict.items():
            value['exercise_date'] = key
            value_list.append(value)
        Price_df = pd.concat(value_list).sort('exercise_date')

        value_list = []
        for key, value in impVol_dict.items():
            value['exercise_date'] = key
            value_list.append(value)
        impVol_df = pd.concat(value_list).sort('exercise_date')

        return Contract_df, Price_df, impVol_df, key_list

    return Contract_dict, Price_dict, impVol_dict, key_list

# 获取数据
date=datetime.date(2018,1,10)
S0=get_price('510050.XSHG',start_date=date,end_date=date,fields=['close']).values[0][0]
Contract,Price,impVol,key_list=get_50ETF_option(date,True)

# 计算残差
Price['T']=np.array([i.days for i in Price['exercise_date']-date])/365
Price['resid']=Price['Call']-Price['Put']-S0+np.array(Contract.index)*exp(Price['T']*(-0.03))
# 排序
try:
    del Contract['exercise_date']
except:
    pass
Mydf=pd.concat([Contract,Price],axis=1)
Mydf=Mydf.sort(['exercise_date','resid'])
#筛选合约
cost=0.005
for key in key_list:
    temp=Mydf[Mydf['exercise_date']==key]
    if temp.iloc[-1,-1]-temp.iloc[0,-1]>cost:
        print(date,':','long {0},short {1},short {2},long {3}'\
              .format(temp.iloc[0,0],temp.iloc[0,1],temp.iloc[-1,0],temp.iloc[-1,1]))



