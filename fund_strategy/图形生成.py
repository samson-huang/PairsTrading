import sys
sys.path.append("c://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//foundation_tools//")
from send_mail_tool import *
from technical_analysis_patterns import (rolling_patterns2pool,plot_patterns_chart,rolling_patterns)
from typing import (List, Tuple, Dict, Callable, Union)
from tqdm.notebook import tqdm
import json
import pandas as pd
import numpy as np
import empyrical as ep
import seaborn as sns
import matplotlib as mpl
import mplfinance as mpf
import matplotlib.pyplot as plt
import datetime
import os


test_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//close.pkl')
test_pre_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//pre_close.pkl')
test_high = pd.read_pickle('C://temp//fund_data//base_data//mkt//high.pkl')
test_low = pd.read_pickle('C://temp//fund_data//base_data//mkt//low.pkl')
test_amount = pd.read_pickle('C://temp//fund_data//base_data//mkt//amount.pkl')
test_open = pd.read_pickle('C://temp//fund_data//base_data//mkt//open.pkl')


def mkdir(path):
   '''
   创建指定的文件夹
   :param path: 文件夹路径，字符串格式
   :return: True(新建成功) or False(文件夹已存在，新建失败)
   '''
   # 去除首位空格
   path = path.strip()
   # 去除尾部 \ 符号
   path = path.rstrip("\\")

   # 判断路径是否存在
   # 存在     True
   # 不存在   False
   isExists = os.path.exists(path)

   # 判断结果
   if not isExists:
      # 如果不存在则创建目录
      # 创建目录操作函数
      os.makedirs(path)
      print(path + ' 创建成功')
      return True
   else:
      # 如果目录存在则不创建，并提示目录已存在
      print(path + ' 目录已存在')
      return False



if __name__ == '__main__':

   # list_sh     上证380   上证180       上证50      沪深300     科创50
   list_sh = ( '000009.SH','000010.SH', '000016.SH', '000300.SH', '000688.SH',
              # 中证1000     中证100   中证500	   中证800
                 '000852.SH', '000903.SH', '000905.SH', '000906.SH')
   # 深圳指数   深证成指    中小板指	   创业板指     深证100
   list_sz = ('399001.SZ', '399005.SZ', '399006.SZ', '399330.SZ')

   list_1 = list_sh + list_sz
   #list_1 = ('000300.SH',)
   list_1 = list(list_1)
   local_datetime = datetime.datetime.now().strftime('%Y%m%d')
   mkdir('C://temp//upload//' + local_datetime + '_pattern_graph//')
   with open('C://temp//upload//codefundsecname.json') as file:
      code2secname = json.loads(file.read())
   #########################生成图片#####################################

   for index_code in list_1:

      local_url='C://temp//upload//'+ local_datetime + '_pattern_graph//'+local_datetime+'_'+code2secname[index_code]+'_detail.jpg'
      #'close,pre_close,high,low,amount'
      #fields=['trade_date','open', 'close', 'low', 'high']

      dfs = [test_open[index_code],test_close[index_code],test_low[index_code],test_high[index_code]]
      result = pd.concat(dfs,axis=1)
      result.columns = ['open', 'close', 'low', 'high']
      
      data1=result[-40:]
      data1.index = pd.to_datetime(data1.index)
      data1.sort_index(inplace=True)
      ###为了提前生成图形，生成t+1天模拟数据，跟T价格指数相同
      data1.loc[data1.index[-1] + datetime.timedelta(days=1)] = data1.iloc[-1, :]


      #############图形判断###############
      #patterns_record1 = rolling_patterns2pool(data1['close'],n=15)
      patterns_record1 = rolling_patterns(data1['close'], n=12)
      plot_patterns_chart(data1,patterns_record1,True,False,code2secname[index_code],local_url.replace('detail', 'overall'))
      plt.title(code2secname[index_code])
      plot_patterns_chart(data1,patterns_record1,True,True,code2secname[index_code],local_url);
      ####################################

      ###############趋与势模型############

      ####################################

   ######################### 邮件发送#####################################
   local_url_mail = 'C://temp//upload//'+ local_datetime + '_pattern_graph//' + local_datetime
   recer = ["tianfangfang1105@126.com","huangtuo02@163.com", ]
   send_fundmail=send_mail_tool(_recer=recer,local_url=local_url_mail).action_send()

