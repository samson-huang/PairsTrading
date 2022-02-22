


import pyfolio as pf
import pandas as pd
import numpy as np
from datetime import datetime
#%matplotlib inline

fund_160706SZ = pd.read_csv("C:\\funds_data\\160706.SZ.csv", index_col=[0])

fund_160706SZ_end=fund_160706SZ.drop_duplicates(subset=['nav_date'], keep='last', inplace=False)

fund_160706SZ_end=fund_160706SZ_end.loc[:,['nav_date','adj_nav']]
fund_160706SZ_end.set_index('nav_date', inplace=True)
fund_160706SZ_end.index = fund_160706SZ_end.index.astype(str)
fund_160706SZ_end.index=[datetime.strptime(x,'%Y%m%d') for x in fund_160706SZ_end.index]
#reset by index order
fund_160706SZ_end=fund_160706SZ_end.sort_index()

#valley = fund_160706SZ_end.index[fund_160706SZ_end.values.argmin()]
#fund_160706SZ_end = np.argmin(fund_160706SZ_end)
#fund_160706SZ_end = fund_160706SZ_end.index[np.argmin(fund_160706SZ_end)]
returns = fund_160706SZ_end.pct_change()
returns=returns.dropna()
returns.index=returns.index.tz_localize('UTC')

pf.create_returns_tear_sheet(returns['adj_nav'])


import pyfolio as pf
import pandas as pd
#%matplotlib inline
from datetime import datetime
return_ser = pd.read_csv('C:\\funds_data\\return_ser.csv')
return_ser['date'] = pd.to_datetime(return_ser['date'])
return_ser.set_index('date', inplace=True)
#return_ser.index=[datetime.strptime(x,'%Y%m%d') for x in return_ser.index]
pf.create_returns_tear_sheet(return_ser['return'])

self.df["date"] = pd.to_datetime(self.df["date"])




out_of_sample = returns.index[-50]
#returns=returns.tz_localize('UTC')
#returns=returns.dropna()
#returns[3500:].describe()
#test=returns['adj_nav'][3900:]
test=returns['adj_nav'].tz_localize('UTC')

pf.create_bayesian_tear_sheet(returns['adj_nav'], live_start_date=out_of_sample)

def get_return(code):
 df = ts.get_k_data(code, start='2010-01-01')
 df.index = pd.to_datetime(df.date)
 return df.close.pct_change().fillna(0).tz_localize('UTC')

test=returns['adj_nav'][3800:]
trace = pf.bayesian.run_model('normal', test)


pf.create_bayesian_tear_sheet(test, live_start_date=out_of_sample)






