


import pyfolio as pf
import pandas as pd
import numpy as np
from datetime import datetime
%matplotlib inline

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
returns=returns[~returns.adj_nav.isin(['NaN'])]

pf.create_returns_tear_sheet(returns['adj_nav'])


import pyfolio as pf
import pandas as pd
%matplotlib inline
 from datetime import datetime
return_ser = pd.read_csv('C:\\funds_data\\return_ser.csv')
return_ser['date'] = pd.to_datetime(return_ser['date'])
return_ser.set_index('date', inplace=True)
#return_ser.index=[datetime.strptime(x,'%Y%m%d') for x in return_ser.index]
pf.create_returns_tear_sheet(return_ser['return'])