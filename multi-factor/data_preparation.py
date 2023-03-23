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

test_circ_mv = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//circ_mv.pkl')
test_close = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//close.pkl')
test_dv_ratio = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//dv_ratio.pkl')
test_dv_ttm = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//dv_ttm.pkl')
test_float_share = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//float_share.pkl')
test_free_share = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//free_share.pkl')

test_pb= pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//pb.pkl')
test_pe = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//pe.pkl')
test_pe_ttm = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//pe_ttm.pkl')
test_ps = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//ps.pkl')
test_total_mv = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//total_mv.pkl')
test_total_share = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//total_share.pkl')

test_turnover_rate= pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//turnover_rate.pkl')
test_turnover_rate_f = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//turnover_rate_f.pkl')
test_volume_ratio = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//volume_ratio.pkl')

#['close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio','pe', 'pe_ttm',
#                          'pb','ps','ps_ttm','dv_ratio','dv_ttm','total_share','float_share','free_share',
#                          'total_mv','circ_mv']

#转换城alphalens-example方式
class transformer_data:
    def __init__(self, start_date='20230320', end_date='20230320'):
        self.start_date = start_date
        self.end_date = end_date
        self.local_url='C://temp//multi_factor_data//base_data//mkt//'
    @staticmethod
    def data_melt(para_name,*args, **kwds):
        para_name_return = pd.read_pickle('C://temp//multi_factor_data//base_data//mkt//'+para_name+'.pkl')
        para_name_return=para_name_return.reset_index()
        para_name_return=para_name_return.melt(id_vars=['trade_date'],var_name='ts_code',value_name=para_name)
        return para_name_return

    @staticmethod
    def data_merge(name_list,*args, **kwds):
        count_num = 1
        for data_name in name_list:
            return_data_indermediate=transformer_data.data_melt(data_name)
            if count_num==1:
                return_data=return_data_indermediate
            else:
               return_data=pd.merge(return_data, return_data_indermediate, on=['trade_date','ts_code'])
            count_num=count_num+1
        return return_data


columns = ['close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio','pe', 'pe_ttm',
                  'pb','ps','ps_ttm','dv_ratio','dv_ttm','total_share','float_share','free_share',
                  'total_mv','circ_mv']
data_final=transformer_data.data_merge(columns)






