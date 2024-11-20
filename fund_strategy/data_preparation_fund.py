import json
import warnings
import time
import akshare as ak
import pandas as pd
import numpy as np




#dir_name = 'C:/Users/huangtuo/.qlib/qlib_data/fund_data/csv'
dir_name = 'C:/Users/huangtuo/.qlib/qlib_data/fund_data/change_csv'

import pandas as pd
import akshare as ak
import random

def fetch_and_save_data(codefundsecname_file, dir_name, start_date, end_date):
    codefundsecname = pd.read_csv(codefundsecname_file)
    lof_code = codefundsecname[codefundsecname['type'] == 'lof']['code']
    etf_code = codefundsecname[codefundsecname['type'] == 'etf']['code']
    index_code = codefundsecname[codefundsecname['type'] == 'index']['code']

    data_codes = {
        'lof_code': lof_code,
        'etf_code': etf_code,
        'index_code': index_code
    }

    for code_type, codes in data_codes.items():
        for original_str in codes:
            try:
                if code_type == 'lof_code':
                    fund_df = ak.fund_lof_hist_em(symbol=original_str[2:], period="daily", start_date=start_date, end_date=end_date, adjust="")
                elif code_type == 'etf_code':
                    fund_df = ak.fund_etf_hist_em(symbol=original_str[2:], period="daily", start_date=start_date, end_date=end_date, adjust="")
                elif code_type == 'index_code':
                    fund_df = ak.stock_zh_index_daily_em(symbol=original_str, start_date=start_date, end_date=end_date)

                if fund_df.empty:
                    raise ValueError(f"No data retrieved for {original_str}")

                fund_df = fund_df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume'
                })
                fund_df = fund_df[['date', 'open', 'close', 'high', 'low', 'volume']]
                fund_df['date'] = pd.to_datetime(fund_df['date']).dt.strftime('%Y-%m-%d')
                fund_df['date'] = fund_df['date'].astype('datetime64[ns]')
                fund_df['code'] = original_str
                file_name = f"{dir_name}/{original_str}.csv"
                fund_df.to_csv(file_name, header=True, index=False)
            except Exception as e:
                print(f"Error for {original_str}: {e}")
            finally:
                # 随机暂停 5 到 15 秒
                random_delay = random.randint(1, 10)
                time.sleep(random_delay)



codefundsecname_file = 'c:\\temp\\upload\\codefundsecname.csv'
#dir_name = 'c:/temp/20240722'
start_date = '20050101'
end_date = '20241118'

fetch_and_save_data(codefundsecname_file, dir_name, start_date, end_date)

#全量替换数据
#C:\qlib-main\scripts
#python dump_bin.py dump_all --csv_path C:\Users\huangtuo\.qlib\qlib_data\fund_data\change_csv --qlib_dir C:\Users\huangtuo\.qlib\qlib_data\fund_data --symbol_field_name code  --date_field_name date  --include_fields open,high,low,close,volume

#增量更新数据

#python dump_bin.py dump_update --csv_path C:\Users\huangtuo\.qlib\qlib_data\fund_data\csv --qlib_dir C:\Users\huangtuo\.qlib\qlib_data\fund_data --symbol_field_name code  --date_field_name date  --include_fields open,high,low,close,volume





#fund_basic = pro.fund_basic(market='E')
#stk_set=list(fund_basic.loc[:, 'ts_code'])
#make_fund_dataset(stk_set, start, end)

'''
from qlib.data import D
instruments = D.instruments(market='test')
data = D.list_instruments(instruments=instruments)
df = pd.DataFrame(columns=['ts_code', 'start', 'end'])
for ts_code, periods in data.items():
    for period in periods:
        start = period[0].floor('D')
        end = period[1].floor('D')
        df = df.append({'ts_code': ts_code,
                        'start': start,
                        'end': end},
                       ignore_index=True)
'''

