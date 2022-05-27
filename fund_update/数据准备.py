import datetime
import json
import os
import pickle
import sys
import warnings
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
import tushare as ts

warnings.filterwarnings('ignore')
# 请根据自己的情况填写ts的token
setting = json.load(open('C://config//config.json'))
# pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)
ts.set_token(setting['token'])
pro = ts.pro_api(timeout=5)

global dataBase
curPath = os.path.abspath(os.path.dirname('c:\\temp\\fund_data\\'))
# rootPath = curPath[:curPath.find("多因子框架\\")+len("多因子框架\\")]
rootPath = curPath
dataBase = rootPath + '\\base_data\\'


def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def update_pickle(text, path):
    with open(path, 'wb') as handle:
        pickle.dump(text, handle)


class DataDownloader:
    def __init__(self, start_date='20050101', end_date=None):
        self.start_date = start_date
        self.end_date = end_date
        self.trade_dates = self.get_trade_dates()
        #self.stk_codes = self.get_stks()
        #获取场内基金代码
        self.fund_codes = self.get_stks_fund()
        # self.template_df = pd.DataFrame(index=self.trade_dates,columns=self.stk_codes)

    def get_trade_dates(self, start_date=None, end_date=None):
        if start_date == None:
            start_date = self.start_date
        end_date = datetime.datetime.now().strftime('%Y%m%d') if end_date == None else self.end_date
        df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
        df[df['is_open'] == 1]['cal_date'].drop_duplicates()
        return df[df['is_open'] == 1]['cal_date'].to_list()

    def get_stks(self):
        stk_set = set()
        for list_status in ['L', 'D', 'P']:
            stk_set |= set(pro.stock_basic(list_status=list_status, fileds='ts_code')['ts_code'].to_list())

        return sorted(list(stk_set))

#########################################################################################
    def get_stks_fund(self):
        #stk_set = DataReader.read_E_fund()
        #赛选特定几个数据
        #list_1 = ('510050.SH', '159902.SZ', '510330.SH', '159915.SZ', '510500.SH', '512010.SH', '502003.SH')
        list_1 = ('000300.SH', '399006.SZ', '399905.SZ')
        list_1 = list(list_1)
        #stk_set = stk_set[stk_set[['ts_code']].apply(lambda x : x.str.contains('|'.join(list_1))).any(1)]
        ###########
        #stk_set = stk_set['ts_code'].to_list()
        stk_set = list_1
        return sorted(list(stk_set))


##########################################################################################

    def get_IdxWeight(self, idx_code):
        """
        指数成分股
        """
        start_date = pd.to_datetime(self.trade_dates[0]) - datetime.timedelta(days=32)
        start_date = start_date.strftime('%Y%m%d')
        trade_dates = self.get_trade_dates(start_date)
        df_ls = []
        while start_date < trade_dates[-1]:
            end_date = pd.to_datetime(start_date) + datetime.timedelta(days=32)
            end_date = end_date.strftime('%Y%m%d')
            raw_df = pro.index_weight(index_code=idx_code, start_date=start_date, end_date=end_date)
            df_ls.append(raw_df.pivot(index='trade_date', columns='con_code', values='weight'))
            start_date = end_date

        res_df = pd.concat(df_ls)
        res_df = res_df[~res_df.index.duplicated(keep='first')]
        res_df = res_df.reindex(trade_dates)
        res_df = res_df.ffill().reindex(self.trade_dates)
        return res_df.sort_index()

    def get_ST_valid(self):
        """
        ST股
        """
        res_df = pd.DataFrame(index=self.trade_dates, columns=self.stk_codes).fillna(1)
        df = pro.namechange(fields='ts_code,name,start_date,end_date')
        df = df[df.name.str.contains('ST')]
        for i in range(df.shape[0]):
            ts_code = df.iloc[i, 0]
            if ts_code not in self.stk_codes:
                continue
            s_date = df.iloc[i, 2]
            e_date = df.iloc[i, 3]
            if e_date == None:
                res_df[ts_code].loc[s_date:] = np.nan
            else:
                res_df[ts_code].loc[s_date:e_date] = np.nan
        return res_df.sort_index()

    def get_suspend_oneDate(self, trade_date, m_ls):
        '''
        tushare的接口一次最多返回5000条数据，所以按天调取。用并行加速。
        '''
        try:
            df = pro.suspend_d(suspend_type='S', trade_date=trade_date)
            m_ls.append([trade_date, df])
        except:
            df = pro.suspend_d(suspend_type='S', trade_date=trade_date)
            m_ls.append([trade_date, df])

    def get_suspend_valid(self):
        '''
        停牌股
        '''
        res_df = pd.DataFrame(index=self.trade_dates, columns=self.stk_codes).fillna(1)

        m_ls = Manager().list()
        pools = Pool(4)
        for date in self.trade_dates:
            pools.apply_async(self.get_suspend_oneDate,
                              args=(date, m_ls)
                              )
        pools.close()
        pools.join()
        m_ls = list(m_ls)
        for date, df in m_ls:
            print(date, df)
            res_df.loc[date, df['ts_code'].to_list()] = np.nan
        return res_df.sort_index()

    def get_limit_oneDate(self, trade_date, m_ls):
        '''
        tushare的接口一次最多返回5000条数据，所以按天调取。用并行加速。
        '''
        try:
            df = pro.limit_list(trade_date=trade_date)
            m_ls.append([trade_date, df])
        except:
            df = pro.suspend_d(trade_date=trade_date)
            m_ls.append([trade_date, df])

    def get_limit_valid(self):
        '''
        停牌股
        '''
        res_df = pd.DataFrame(index=self.trade_dates, columns=self.stk_codes).fillna(1)
        m_ls = Manager().list()
        pools = Pool(3)
        for date in self.trade_dates:
            pools.apply_async(self.get_limit_oneDate,
                              args=(date, m_ls)
                              )
        pools.close()
        pools.join()
        m_ls = list(m_ls)
        for date, df in m_ls:
            res_df.loc[date, df['ts_code'].to_list()] = np.nan
        return res_df.sort_index()

    def get_dailyMkt_oneStock(self, ts_code, m_ls):
        '''
        前复权的行情数据

        因为tushare下载复权行情接口一次只能获取一只股票
        所以使用多进行并行
        '''

        try:
            # 偶尔会因为网络问题请求失败，报错重新请求
            df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=self.start_date, end_date=self.end_date)
            m_ls.append(df)
        except:
            df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=self.start_date, end_date=self.end_date)
            m_ls.append(df)

    def get_dailyMkt_mulP(self):
        m_ls = Manager().list()
        pools = Pool(3)  # 开太多会有访问频率限制
        for ts_code in self.stk_codes:
            pools.apply_async(self.get_dailyMkt_oneStock,
                              args=(ts_code, m_ls))
        pools.close()
        pools.join()
        m_ls = list(m_ls)
        raw_df = pd.concat(m_ls)
        res_dict = {}
        for data_name in ['open', 'close', 'high', 'low', 'vol', 'amount']:
            res_df = raw_df.pivot(index='trade_date', columns='ts_code', values=data_name)
            res_dict[data_name] = res_df.sort_index()
        return res_dict
######################场内基金更新######################################################
    def get_E_fund(self):
        '''
        场内基金数据
        '''
        res_df=pro.index_basic()
        return res_df

    def get_dailyMkt_mulP_fund(self):
        m_ls = Manager().list()
        pools = Pool(3)  # 开太多会有访问频率限制
        for ts_code in self.fund_codes:
            pools.apply_async(self.get_dailyMkt_oneFund,
                              args=(ts_code, m_ls))
        pools.close()
        pools.join()
        m_ls = list(m_ls)
        raw_df = pd.concat(m_ls)
        ##############################################
        #raw_df.drop_duplicates(subset=['ts_code', 'nav_date'], keep='last', inplace=True)
        res_dict = {}
        for data_name in ['open', 'close', 'high', 'low','pre_close', 'vol', 'amount']:
            res_df = raw_df.pivot(index='trade_date', columns='ts_code', values=data_name)
            res_dict[data_name] = res_df.sort_index()
        return res_dict

    def get_dailyMkt_oneFund(self, ts_code, m_ls):
        '''
        前复权的行情数据

        因为tushare下载复权行情接口一次只能获取一只股票
        所以使用多进行并行
        '''

        try:
            # 偶尔会因为网络问题请求失败，报错重新请求
            df = pro.index_daily(ts_code=ts_code, start_date=self.start_date, end_date=self.end_date)
            m_ls.append(df)
        except:
            df = pro.index_daily(ts_code=ts_code, start_date=self.start_date, end_date=self.end_date)
            m_ls.append(df)
class DataWriter:
    @staticmethod
    def commonFunc(data_path, getFunc, cover, *args, **kwds):
        if not os.path.exists(data_path) or cover:
            t1 = datetime.datetime.now()
            print(f'--------{data_path},第一次下载该数据，可能耗时较长')
            newData_df = eval(f'DataDownloader().{getFunc}(*args,**kwds)')
            newData_df.to_pickle(data_path)
            t2 = datetime.datetime.now()
            print(f'--------下载完成,耗时{t2 - t1}')
        else:
            savedData_df = pd.read_pickle(data_path)
            savedLastDate = savedData_df.index[-1]
            print(f'---------{data_path}上次更新至{savedLastDate}，正在更新至最新交易日')

            lastData_df = eval(f'DataDownloader(savedLastDate).{getFunc}(*args,**kwds)')
            newData_df = pd.concat([savedData_df, lastData_df]).sort_index()
            newData_df = newData_df[~newData_df.index.duplicated(keep='first')]
            newData_df.to_pickle(data_path)
            print(f'---------已更新至最新日期{newData_df.index[-1]}')
        newData_df.index = pd.to_datetime(newData_df.index)
        return newData_df

    @staticmethod
    def update_IdxWeight(stk_code, cover=False):
        data_path = dataBase + f'daily/idx_cons/{stk_code}.pkl'
        return DataWriter.commonFunc(data_path, 'get_IdxWeight', cover, stk_code)

    @staticmethod
    def update_ST_valid(cover=False):
        data_path = dataBase + f'daily/valid/ST_valid.pkl'
        return DataWriter.commonFunc(data_path, 'get_ST_valid', cover)

    @staticmethod
    def update_suspend_valid(cover=False):
        data_path = dataBase + 'daily/valid/suspend_valid.pkl'
        return DataWriter.commonFunc(data_path, 'get_suspend_valid', cover)

    @staticmethod
    def update_limit_valid(cover=False):
        data_path = dataBase + 'daily/valid/limit_valid.pkl'
        return DataWriter.commonFunc(data_path, 'get_limit_valid', cover)

    @staticmethod
    def update_dailyMkt(cover=False):
        '''
            需要保证已存储的ochlv数据的日期一致
        '''
        if not os.path.exists(dataBase + f'daily/mkt/open.pkl') or cover:
            print(f'--------Mkt,第一次下载该数据，可能耗时较长')
            res_dict = DataDownloader().get_dailyMkt_mulP()
            for data_name, df in res_dict.items():
                data_path = dataBase + f'daily/mkt//{data_name}.pkl'
                df.to_pickle(data_path)
        else:
            savedData_df = pd.read_pickle(dataBase + f'daily/mkt/{data_name}.pkl')
            savedLastDate = savedData_df.index[-1]
            print(f'---------Mkt,上次更新至{savedLastDate}，正在更新至最新交易日')

            res_dict = DataDownloader(savedLastDate).get_dailyMkt_mulP()
            new_df = pd.DataFrame()
            for data_name, last_df in res_dict.items():
                data_path = dataBase + f'daily/mkt//{data_name}.pkl'
                new_df = pd.concat([savedData_df, last_df]).sort_index()
                new_df = new_df[~new_df.index.duplicated(keep='first')]
                new_df.to_pickle(data_path)
            print(f'---------已更新至最新日期{new_df.index[-1]}')

###########################更新场内基金#################################################
    @staticmethod
    def update_E_fund(cover=False):
        data_path = dataBase + 'all_e_funds.pkl'
        return DataWriter.commonFunc(data_path, 'get_E_fund', cover)

    @staticmethod
    def update_dailyMkt_fund(cover=False):
        '''
            需要保证已存储的ochlv数据的日期一致
        '''
        if not os.path.exists(dataBase + 'mkt//open.pkl') or cover:
            print(f'--------Mkt,第一次下载该数据，可能耗时较长')
            res_dict = DataDownloader().get_dailyMkt_mulP_fund()
            for data_name, df in res_dict.items():
                data_path = dataBase + f'mkt//{data_name}.pkl'
                df.to_pickle(data_path)
        else:
            savedData_df = pd.read_pickle(dataBase + f'mkt//open.pkl')
            savedLastDate = savedData_df.index[-1]
            print(f'---------Mkt,上次更新至{savedLastDate}，正在更新至最新交易日')

            res_dict = DataDownloader(savedLastDate).get_dailyMkt_mulP_fund()
            new_df = pd.DataFrame()
            for data_name, last_df in res_dict.items():
                data_path = dataBase + f'mkt//{data_name}.pkl'
                savedData_df1 = pd.read_pickle(dataBase + f'mkt//{data_name}.pkl')
                new_df = pd.concat([savedData_df1, last_df]).sort_index()
                new_df = new_df[~new_df.index.duplicated(keep='first')]
                new_df.to_pickle(data_path)
            print(f'---------已更新至最新日期{new_df.index[-1]}')

class DataReader:
    @staticmethod
    def commonFunc(data_path):
        if not os.path.exists(data_path):
            print(f'{data_path}不存在，请先调用DataWriter().update_xx')
            return
        df = pd.read_pickle(data_path)
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def read_IdxWeight(stk_code):
        data_path = dataBase + f'daily/idx_cons/{stk_code}.pkl'
        return DataReader.commonFunc(data_path)

    @staticmethod
    def read_ST_valid():
        data_path = dataBase + f'daily/valid/ST_valid.pkl'
        return DataReader.commonFunc(data_path)

    @staticmethod
    def read_suspend_valid():
        data_path = dataBase + 'daily/valid/suspend_valid.pkl'
        return DataReader.commonFunc(data_path)

    @staticmethod
    def read_limit_valid():
        data_path = dataBase + 'daily/valid/limit_valid.pkl'
        return DataReader.commonFunc(data_path)

    @staticmethod
    def read_dailyMkt(data_name):
        data_path = dataBase + f'mkt//{data_name}.pkl'
        return DataReader.commonFunc(data_path)

    @staticmethod
    def read_index_dailyRtn(index_code, start_date='20100101'):
        df = pro.index_daily(ts_code=index_code, start_date=start_date).set_index('trade_date').sort_index()
        df.index = pd.to_datetime(df.index)
        return df['pct_chg'] / 100

    @staticmethod
    def read_dailyRtn():
        df = DataReader.read_dailyMkt('close')
        return df.pct_change()

##########################读取场内基金数据#################################################
    @staticmethod
    def read_E_fund():
        data_path = dataBase + 'all_e_funds.pkl'
        return DataReader.commonFunc(data_path)

if __name__ == '__main__':
    #DataWriter.update_E_fund(cover=False)
    DataWriter.update_dailyMkt_fund(cover=False)




###########################################################################
    #DataWriter.update_ST_valid(cover=True)
    #DataWriter.update_suspend_valid(cover=True)
    #DataWriter.update_IdxWeight('399300.SZ', cover=True)
    #DataWriter.update_dailyMkt(cover=True)
    #DataWriter.update_limit_valid(cover=True)



'''
test_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//close.pkl')
test_pre_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//pre_close.pkl')
test_high = pd.read_pickle('C://temp//fund_data//base_data//mkt//high.pkl')
test_low = pd.read_pickle('C://temp//fund_data//base_data//mkt//low.pkl')
test_amount = pd.read_pickle('C://temp//fund_data//base_data//mkt//amount.pkl')


index_code = '000300.SH'
#'close,pre_close,high,low,amount'
dfs = [test_close[index_code],test_pre_close[index_code],test_high[index_code],test_low[index_code],test_amount[index_code]]
result = pd.concat(dfs,axis=1)
result.columns = ['close','pre_close','high','low','amount']


result = result.dropna(inplace=False) 

#result.to_csv('c://temp//000300SH.csv')
#import datetime as datetime
#close_df=pd.read_csv('c://temp//000300SH.csv')
#close_df['trade_date']=pd.to_datetime(close_df['trade_date'].astype(str))
#close_df.set_index('trade_date', inplace=True)
result.index=pd.to_datetime(result.index)
result.sort_index(inplace=True) 


close_df=result

price_df=close_df[['close']]
# 指标计算
LR = cala_LR(price_df['close'])


rsrs = RSRS_improve2()  # 调用RSRS计算类
signal_df = rsrs.get_RSRS(close_df, (1 - LR), 10, 60, 'ols')  # 获取各RSRS信号

signal_df.tail()

'''
