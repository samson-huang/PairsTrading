from datetime import datetime,timedelta

######################################### 公用 ####################################################
# 绕过查询限制沪深港通资金流向
#moneyflow_hsgt
# ts的日历需要处理一下才会返回成交日列表
## 减少ts调用 改用jq的数据....
def query_trade_dates(start_date: str, end_date: str) -> list:
    start_date = parse(start_date).strftime('%Y-%m-%d')
    end_date = parse(end_date).strftime('%Y-%m-%d')
    ############################
    df = my_pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
    dates = df.query('is_open==1')['cal_date'].values.tolist()
    return  dates
    #return get_trade_days(start_date, end_date).tolist()
####################################################################
def GetEveryDay(begin_date,end_date):
#获得两日期间的日期列表
    global date_list
    date_list = []
    begin_date = datetime.strptime(begin_date,"%Y%m%d")
    end_date = datetime.strptime(end_date,"%Y%m%d")
    while begin_date <= end_date:
          date_str = begin_date.strftime("%Y%m%d")
          date_list.append(date_str)
          begin_date += datetime.timedelta(days=1)
    #print('日期列表已形成')
    return date_list

###################################################################
def distributed_other_query(query_func_name,
                      start_date,
                      end_date,
                      limit=300):
                      	
    dates = GetEveryDay(start_date,end)
    n_days = len(dates)

    if  n_days > limit:
        n = limit 

        df_list = []
        i = 0
        pos1, pos2 = n * i, n * (i + 1) - 1

        while pos2 < n_days:
            df = query_func_name(
                start_date=dates[pos1],
                end_date=dates[pos2])
            df_list.append(df)
            i += 1
            pos1, pos2 = n * i, n * (i + 1) - 1
        if pos1 < n_days:
            df = query_func_name(
                start_date=dates[pos1],
                end_date=dates[-1])
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
    else:
        df = query_func_name(
            start_date=start_date,
            end_date=end_date)
    return df


start = '20141117'
end = '20200827'
daily_northmoney = distributed_other_query(my_pro.moneyflow_hsgt,
                                   start,
                                   end)
                                   
                                   
test_t=daily_northmoney
daily_northmoney.index=daily_northmoney['trade_date']
daily_northmoney=daily_northmoney[['trade_date','north_money']]
daily_northmoney=daily_northmoney.sort_index()
net_flow=daily_northmoney   



index_df = my_pro.query('index_daily', ts_code='000001.SH', 
start_date=start, end_date=end,fields='trade_date,close') 
index_df.index=index_df['trade_date']
index_df=pd.DataFrame(index_df['close'])
index_df=index_df.sort_index()  



df=pd.merge(index_df,net_flow,left_index=True,right_index=True,how='outer')
df=df[['close','north_money']]
df=df.reset_index()
df.dropna(axis=1,how='any')
df.columns = [ 'date','close','north_money'] 

line = df.plot.line(x='date', y='close', secondary_y=True,
                    color='r', figsize=(18, 8), title='北向资金历史日频资金流')
df.plot.bar(x='date', y='north_money', ax=getattr(
    line, 'left_ax', line), color='DeepSkyBlue', rot=90)


xticks = list(range(0, len(df), 50))

xlabels = df['date'].loc[xticks].values.tolist()

plt.xticks(xticks, xlabels, rotation=90);


                              