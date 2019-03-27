import tushare as ts
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#import cvxopt as opt
#from cvxopt import blas, solvers
import pandas as pd


ts.get_sz50s()

sz50s=ts.get_sz50s()

#convert np.array
sz50s_code=sz50s["code"].values
convertible_bond_code=(['300059.sz','123006.sz'])


#######################test阶段##########################
####################从wind取数据#######################################
from WindPy import *
w.start()
wsddata1=w.wsd('123006.sz', "open,high,low,close,volume,amt",'20190101','20190301', "Fill=Previous")
wsddata1.Data





# 取数据的命令如何写可以用命令生成器来辅助完成
wsd_data=w.wsd("123006.sz", "open,high,low,close", "2019-01-01", "2019-03-01", "Fill=Previous")

#演示如何将api返回的数据装入Pandas的Series
open=pd.Series(wsd_data.Data[0])
high=pd.Series(wsd_data.Data[1])
low=pd.Series(wsd_data.Data[2])
close=pd.Series(wsd_data.Data[3])

#print('open:/n',open)
#print('high:/n',high)
#print('low:/n',low)
#print('close:/n',close)

#演示如何将api返回的数据装入Pandas的DataFrame
fm=pd.DataFrame(wsd_data.Data,index=wsd_data.Fields,columns=wsd_data.Times)
fm=fm.T #将矩阵转置
ss123006=fm
print('fm:/n',fm)

###
df.to_csv ("testfoo.csv" , encoding = "utf-8")
'''需要写clas 封装取数据的过程
def output_data(source,): 
	  if source 
'''
##################################################################################
############################第一步###############################################
########################################################################################
def output_data(security,source,begin_date,end_date,column): 
	  if source=='wind':
	     wsd_data=w.wsd(security,column, begin_date,end_date, "Fill=Previous")
	     fm=pd.DataFrame(wsd_data.Data,index=wsd_data.Fields,columns=wsd_data.Times)
	  return(fm.T) 


convertible_bond_code=(['300059.sz','123006.sz'])
symbols= convertible_bond_code
column= "open,high,low,close,volume"	   
pnls2 = {i:output_data(i,'wind',"2019-01-01","2019-01-31",column) for i in symbols}
	
	
	
###############################
###########################plot########################
# solve  chinese dislay
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plot them
plt.plot(pnls2['300059.sz']['CLOSE'], label='东方财富')
plt.plot(pnls2['123006.sz']['CLOSE'], label='东财转债')
	
	
# generate a legend box
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
       ncol=4, mode="expand", borderaxespad=0.)
 
# annotate an important value
#plt.annotate("Important value", (55,20), xycoords='data',
#         xytext=(5, 38),
#         arrowprops=dict(arrowstyle='->'))
plt.show()	


##############difference#############
def total_data(data_p,symbols_code):
    # for modify
    symbols_func=symbols_code[1:len(symbols_code)]
    total_data =pd.DataFrame([data_p[symbols_code[0]]['CLOSE'].sort_index().pct_change()])
    for i in symbols_func:       
       total_data= total_data.append(pd.DataFrame([data_p[i]['CLOSE'].sort_index().pct_change()]))
    return(total_data)
    
    ################################################

total_data1=total_data(pnls2,convertible_bond_code)

total_data1.index=convertible_bond_code
#去掉na值
total_data1=total_data1.dropna(axis=1,how='all') 
###################plot#############################




plt.rcParams['figure.figsize'] = (10.0, 4.0) 
plt.plot(total_data1.T, alpha=.4);
plt.xlabel('time')
plt.ylabel('returns')
####################转债折溢价######################
###########(可转债价格/（100/转股价）)/正股股价
###########(可转债价格*转股价*0.01)/正股股价
def conversion_data(data_p,symbols_code):
    # for modify
    symbols_func=symbols_code[1:len(symbols_code)]
    total_data =pd.DataFrame([(data_p[symbols_code[1]]['CLOSE']*0.1136)/data_p[symbols_code[0]]['CLOSE']])

    return(total_data)


conversion_data1=conversion_data(pnls2,convertible_bond_code)

##################################################

conversion_data1.index=['conversion']
#去掉na值
conversion_data1=conversion_data1.dropna(axis=1,how='all') 
###################plot#############################




plt.rcParams['figure.figsize'] = (10.0, 4.0) 
plt.plot(conversion_data1.T, alpha=.4);
plt.plot()
plt.xlabel('time')
plt.ylabel('returns')
###############################################
###########################plot########################
#x=conversion_data1.T.index

#y=np.random.rand(len(conversion_data1.T.index))*0+
#plt.plot(x,y)	

#################dataframe转list############
conv_min=min(conversion_data1.T.index)
conv_max=max(conversion_data1.T.index)
conv_mean=np.array(conversion_data1).mean()
conv_std=np.array(conversion_data1).std()
# solve  chinese dislay
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plot them
plt.plot(conversion_data1.T, alpha=.4)

plt.hlines(conv_mean,conv_min,conv_max)
plt.hlines(conv_mean-conv_std,conv_min,conv_max,colors = "c", linestyles = "dashed")
plt.hlines(conv_mean+conv_std,conv_min,conv_max,colors = "c", linestyles = "dashed")
  
# generate a legend box
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
       ncol=4, mode="expand", borderaxespad=0.)
 
# annotate an important value
#plt.annotate("Important value", (55,20), xycoords='data',
#         xytext=(5, 38),
#         arrowprops=dict(arrowstyle='->'))
plt.show()	



#################################################################
#######################################
# e.g.,  findNextPosition(r)
#        findNextPosition(r, 1174)
# Check they are increasing and correctly offset
def findNextPosition(ratio, startDay = 1, k = 1):
    m = ratio.mean()
    s = ratio.std()
    up = m + k *s
    down = m - k *s

    if(startDay > 1):
      ratio = ratio[0][startDay-1:]
    
    isExtreme = ratio >= up | ratio <= down
  
    #if(!any(isExtreme))
       #return(integer())

    start = which(isExtreme)[1]
    backToNormal = if(ratio[0][start] > up)
                      ratio[ - (1:start) ] <= m
                    else
                     ratio[ - (1:start) ] >= m

   # return either the end of the position or the index 
   # of the end of the vector.
   # Could return NA for not ended, i.e. which(backToNormal)[1]
   # for both cases. But then the caller has to interpret that.
   
    end = if(any(backToNormal))
             which(backToNormal)[1] + start
          else
             length(ratio)
  
    return(c(start, end) + startDay - 1) 

#############################################
def getPositions(ratio, k = 1, m = ratio.mean(), s = ratio.std()):

    {
       ##when = list()
       cur = 1
    
       while(cur < length(ratio)) {
          tmp = findNextPosition(ratio, cur, k, m, s)
          if(length(tmp) == 0)  # done
             break
          when[[length(when) + 1]] = tmp
          if(is.na(tmp[2]) || tmp[2] == length(ratio))
             break
          cur = tmp[2]
        }
    
       return(cur)
    }
######################################################
# test few data
symbols= convertible_bond_code 
#symbols= ['GOOG']  
#pnls1 = {i:dreader.DataReader(i,'yahoo','2019-01-01','2019-03-01') for i in symbols}

pnls2 = {i:ts.get_hist_data(i,start='2019-01-01',end='2019-03-01') for i in symbols}

# for modify
for i in symbols:        # 第二个实例
   pnls2[i]['close'].index = pnls2[i]['close'].index.astype('datetime64[ns]')
