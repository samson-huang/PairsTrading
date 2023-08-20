import qlib
from qlib.constant import REG_CN
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data import D

instruments = D.instruments(market='all_fund')
data = D.list_instruments(instruments=instruments)

all_fund = pd.DataFrame(columns=['ts_code', 'start', 'end'])
for ts_code, periods in data.items():
    for period in periods:
        start = period[0].floor('D')
        end = period[1].floor('D')
        all_fund = all_fund.append({'ts_code': ts_code,
                        'start': start,
                        'end': end},
                       ignore_index=True)


fund_basic=pd.read_csv('c:\\temp\\fund_basic_new.csv')
fund_basic = pro.fund_basic(market='E')
stock_fund_basic = fund_basic.query('invest_type in ("被动指数型") and fund_type =="股票型"')
bond_fund_basic = fund_basic.query('invest_type in ("被动指数型") and fund_type =="债券型"')
commodity_fund_basic = fund_basic.query('fund_type =="商品型"')

new_fund_basic = pd.concat([stock_fund_basic, bond_fund_basic, commodity_fund_basic])

new_fund_basic['ts_code'] = new_fund_basic['ts_code'].apply(lambda x: x.split('.')[1] + x.split('.')[0])

common_ids = set(new_fund_basic['ts_code']) & set(all_fund['ts_code'])
new_all_fund = all_fund[all_fund['ts_code'].isin(common_ids)]

with open('C:/Users/huangtuo/.qlib/qlib_data/fund_data/instruments/pond_fund.txt', 'w') as f:
    new_all_fund.to_csv(f,sep='	', header=False, index=False)#一个空格折腾很久







