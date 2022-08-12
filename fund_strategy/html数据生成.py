pro.index_daily(ts_code='000009.SH', start_date='20220812', end_date='20220812')

test_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//close.pkl')
test_pre_close = pd.read_pickle('C://temp//fund_data//base_data//mkt//pre_close.pkl')
test_high = pd.read_pickle('C://temp//fund_data//base_data//mkt//high.pkl')
test_low = pd.read_pickle('C://temp//fund_data//base_data//mkt//low.pkl')
test_amount = pd.read_pickle('C://temp//fund_data//base_data//mkt//amount.pkl')
test_open = pd.read_pickle('C://temp//fund_data//base_data//mkt//open.pkl')

test_open1 = test_open.stack()
test_open1 = test_open1.reset_index()
test_open1.columns = ['trade_date', 'symbol', 'OPEN']

test_high1 = test_high.stack()
test_high1 = test_high1.reset_index()
test_high1.columns = ['trade_date', 'symbol', 'HIGH']

test_low1 = test_low.stack()
test_low1 = test_low1.reset_index()
test_low1.columns = ['trade_date', 'symbol', 'LOW']

test_close1 = test_close.stack()
test_close1 = test_close1.reset_index()
test_close1.columns = ['trade_date', 'symbol', 'CLOSE']

test1 = pd.merge(test_open1,test_high1,on=["trade_date","symbol"])
test2 = pd.merge(test_low1,test_close1,on=["trade_date","symbol"])
test3 = pd.merge(test1,test2,on=["trade_date","symbol"])

test3.index = pd.to_datetime(test3['trade_date'])
del test3['trade_date']

#OPEN	HIGH	LOW	CLOSE	symbol
test_open1
test_high1.head()