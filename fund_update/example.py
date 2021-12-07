test123=pd.read_excel("C:\\funds_data\\src\\tradedays.xlsx", index_col=[0])
fund123=pd.read_excel("C:\\funds_data\\src\\all_funds.xlsx", index_col=[0])
fund123[fund123.delist_date.isin(['NaN'])]
delist_date
fund123[(fund123['delist_date']='NaN') & (fund123['due_date']='NaN')] 
fund123[fund123.delist_date.isin(['NaN'])&fund123.due_date.isin(['NaN'])].head()
fund5=fund123[fund123.delist_date.isin(['NaN'])&fund123.due_date.isin(['NaN'])]
fund3=fund123[fund123.delist_date.isin(['NaN'])&fund123.due_date.isin(['NaN'])].tail()

fund4=fund3['ts_code']
fund123.groupby('invest_type').count()
fund123.groupby(['invest_type','fund_type']).count()
fund123[fund123.invest_type.isin([''])]
df[~df.isin(filter_condition)["a"]]
test3=test123((test123['is_open']=='SSE')&(test123['is_open']=='SSE'))
now = datetime.now()
nowday= now.strftime("%Y%m%d")
test123[test123['is_open']==1]&test123[test123['is_open']<=int(nowday)]


test3=test123[(test123["is_open"] == 1) & (test123['cal_date']<=int(nowday))].tail()
test4=test3['cal_date'].apply(str) 
#str4=test4.str.cat(sep=',')


fund4=fund3['ts_code'].apply(str) 
fund4=fund4.str.cat(sep=',')

for date in test4:
	fund_v = pro.fund_nav(ts_code=fund4,nav_date=date)
	fund_v.to_csv(os.path.join('C:\\factors_data\\daily_data\\', date+'.csv'))








fund5=fund123[fund123.delist_date.isin(['NaN'])&fund123.due_date.isin(['NaN'])]
fund5=fund5['ts_code'].apply(str)
#fund5=fund5.str.cat(sep=',')
fund_v = pro.fund_nav(ts_code=fund5,nav_date='20211130')

for date in fund5[0:i]:
	fund_v = pro.fund_nav(ts_code=fund5,nav_date='20211130')



	
data1=pro.fund_nav(nav_date='99991212')	
i=0
fund_count=fund5.count()
while (i <fund_count):
  if i+90>=fund_count:
    data2=pro.fund_nav(ts_code=fund5[i:fund_count].str.cat(sep=','),nav_date='20211130')
  else: 
    data2=pro.fund_nav(ts_code=fund5[i:i+90].str.cat(sep=','),nav_date='20211130')
  i=i+90
  data1=pd.concat([data1, data2], axis=0)
  print(fund_temp)
  time.sleep(1)
data1.to_csv(os.path.join('C:\\factors_data\\daily_data\\', '20211130'+'.csv'))  
  
df = pro.fund_nav


import datacompy
fund_com=fund123[fund123.delist_date.isin(['NaN'])&fund123.due_date.isin(['NaN'])]
daily_fund=pd.read_csv("C:\\factors_data\\daily_data\\20211130.csv", index_col=[0])

compare = datacompy.Compare(fund_com,daily_fund, join_columns='ts_code')
print(compare.matches())
print(compare.report())