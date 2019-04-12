import tushare as ts
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#import cvxopt as opt
#from cvxopt import blas, solvers
import pandas as pd
import random

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
#可以从本地取数据
#from WindPy import *
#w.start()
convertible_bond_code=(['300059.sz','123006.sz'])



def output_data(security,source,begin_date,end_date,column): 
	  if source=='wind':
	     wsd_data=w.wsd(security,column, begin_date,end_date, "Fill=Previous")
	     fm=pd.DataFrame(wsd_data.Data,index=wsd_data.Fields,columns=wsd_data.Times)
	  return(fm.T) 


convertible_bond_code=(['300059.sz','123006.sz'])
symbols= convertible_bond_code
column= "open,high,low,close,volume"	   
pnls2 = {i:output_data(i,'wind',"2019-01-01","2019-04-10",column) for i in symbols}
	
pnls2['300059.sz'].to_csv ("C:/quants/wind_api/sz300059.csv" , encoding = "utf-8")	
pnls2['123006.sz'].to_csv ("C:/quants/wind_api/sz123006.csv" , encoding = "utf-8")	


#########################直接导入本地csv文件###################################

sz300059=pd.read_csv("C:/quants/wind_api/sz300059.csv",index_col=0 , encoding = "utf-8")
sz123006=pd.read_csv("C:/quants/wind_api/sz123006.csv",index_col=0 , encoding = "utf-8")
###############
sz300059=pd.read_csv("D:/quant/python/PairsTrading-master/sz300059.csv",index_col=0 , encoding = "utf-8")
sz123006=pd.read_csv("D:/quant/python/PairsTrading-master/sz123006.csv",index_col=0 , encoding = "utf-8")
####################
pnls2 = {'300059.sz': sz300059,  '123006.sz':sz123006}
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

plt.plot(total_data1.T)





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
#x=conversion_data1.T.index

#y=np.random.rand(len(conversion_data1.T.index))*0+
#plt.plot(x,y)	





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
      ratio = ratio[startDay:]
    #if(startDay > 1):
      #ratio = ratio[0][startDay-1:]
    
    #isExtreme = ratio >= up | ratio <= down
    isExtreme =np.bitwise_xor(ratio >= up,ratio <= down)
    
    if(any(isExtreme)):
    	 1==1
    else:
       return list()


    #x_data[x_data == '?'] = 0
    #start = which(isExtreme)[1]
    start =np.where(isExtreme==1)[0][0]
    if(ratio[start]>up):
       backToNormal =ratio[start:] <= m 
    else:
       backToNormal =ratio[start:] >= m

   # return either the end of the position or the index 
   # of the end of the vector.
   # Could return NA for not ended, i.e. which(backToNormal)[1]
   # for both cases. But then the caller has to interpret that.
   
    if(any(backToNormal)):
       end =np.where(backToNormal==1)[0][0]+ start
    else:
       end =len(ratio)-1
    if(startDay > 1): 
       return(np.array([start,end]) + startDay) 
    else:
       return(np.array([start,end])) 


k = 1
r=np.array(conversion_data1.iloc[0])
a = findNextPosition(r,1, k = k)

b = findNextPosition(r, a[1], k = k)

c = findNextPosition(r, b[1], k = k)

d = findNextPosition(r, c[1], k = k)
e=findNextPosition(r, d[1], k = k)
print(a,b,c,d,e)
print(d,sz300059.index[d], r[d])
#############################################
#################dataframe转list############
def  plotRatio(conversion_data1):
     conv_min=min(conversion_data1)
     conv_max=max(conversion_data1)
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

#############################################
########################################
def  showPosition(pos,stock_index,stock_close):
     cur = 0
     color=['b','c','g','k','m','r','y']
     marker =['*','+','o','>','x','<']
     while(cur < len(pos)):
     	
     	plt.scatter(stock_index.index[pos[cur]], stock_close[pos[cur]], 
     	 color=color[random.randint(0,6)], marker=marker[random.randint(0,5)])
     	cur=cur+1     
#######################################
###################################################################
def getPositions(ratio, k = 1):


       m = ratio.mean()
       s = ratio.std()
       when = list()
       cur = 1
    
       while(cur < len(ratio)):
          tmp = findNextPosition(ratio, cur, k)
          #print(str(cur)+"   ")
          if(len(tmp)<1): 
             break
          #when[(len(when) + 1):]= tmp
          when.append(tmp)
          if(np.isnan(tmp[1]) and tmp[1] == len(ratio)):
             break
          cur = tmp[1]
        
    
       return(when)
############################################
pos = getPositions(r, k)
######################################################

##############图像展示所有的点###########################
plotRatio(conversion_data1)
showPosition(pos,sz300059,r)
'''
plt.scatter(sz300059.index[a], r[a], color='r', marker='+')
plt.scatter(sz300059.index[b], r[b], color='b', marker='o')
plt.scatter(sz300059.index[c], r[c], color='y', marker='*')
plt.scatter(sz300059.index[d], r[d], color='y', marker='>')
'''
#plt.scatter(sz300059.index[e], r[e], color='y', marker='<')
#################################################
plt.show()
#########################################################
#############################################################
##################第一阶段完成############################
                 



################第二阶段###############
##############盈利计算函数########### 
############apply函数只能用在dataframe类型下############
#pos_pd=pd.DataFrame(pos)
     
def positionProfit(pos, stockPriceA, stockPriceB,ratioMean,p = .001):
    byStock = bool(0)
    priceA = stockPriceA[np.array(pos).ravel().tolist()]
    priceB = stockPriceB[np.array(pos).ravel().tolist()]

    if(type(pos)==list):
       #ans=apply(pos, positionProfit,stockPriceA, stockPriceB, ratioMean, p)
       #pos_t=pd.DataFrame(pos).T
       cur=0
       ans=[]
       while(cur < len(pos)):
             ans.append(positionProfit(pos[cur],stockPriceA,stockPriceB,ratioMean))    
             cur=cur+1 
       #ans=pos_t.apply(positionProfit,args=(stockPriceA, stockPriceB,ratioMean,p,))
       if(byStock):
          #rownames(ans) = c("A", "B", "commission")
          ans.columns= c("A", "B", "commission")
       return(ans)
      # how many units can we by of A and B with $1
    unitsOfA = 1/priceA.tolist()[0]
    unitsOfB = 1/priceB.tolist()[0]
    
      # The dollar amount of how many units we would buy of A and B
      # at the cost at the end of the position of each.
    amt = [unitsOfA * priceA.tolist()[1], unitsOfB * priceB.tolist()[1]]
    
      # Which stock are we selling
    if(priceA.tolist()[1]/priceB.tolist()[1] > ratioMean):
       sellWhat = "A"
    else: 
       sellWhat = "B"
    
    if(sellWhat == "A"):
       profit = [(1 - amt[0]),  (amt[1] - 1), - p * sum(amt)]
    else: 
       profit = [(1 - amt[1]),  (amt[0] - 1),  - p * sum(amt)]
    
    if(byStock):
       return(profit)
    else:
     return(sum(profit))

#######################################                 
r.mean()     
pos = getPositions(r, k)
prof = positionProfit(pos,pnls2['300059.sz']['CLOSE'],pnls2['123006.sz']['CLOSE'],r.mean())
         
                 
 ########################################
 ###############第三阶段测试K值的取值###################                
#alist = np.random.rand(range(r.min(),r.max()),10) 
alist=[]
random = np.random.RandomState(0)#RandomState生成随机数种子
for i in range(100):#随机数个数
    a = random.uniform(r.min(),r.max())#随机数范围
    alist.append(round(a,5))#随机数精度要求


#################按随机生成的K测试盈利分布############################
profits
profits=[]
cur=0
while(cur < len(alist)):
      profits.append(sum(positionProfit(getPositions(r,alist[cur]),
        pnls2['300059.sz']['CLOSE'],pnls2['123006.sz']['CLOSE'],r.mean())))    
      cur=cur+1 







##############################################################
########################第四阶段 生成下单##################
######在可转债上的图像#############
r=np.array(conversion_data1.iloc[0])
plotRatio(conversion_data1)

showPosition(pos,sz300059,r)

plt.show()

#plt.plot(sz300059, label='东方财富')
plotRatio(sz300059['CLOSE'])

showPosition(pos,sz300059,sz300059['CLOSE'])

plt.show()



#####################
pos = getPositions(r, k)
test1.index[np.array(pos).ravel().tolist()]
with open('c://excel-export//write.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for list in data:
        print(list)
        csv_writer.writerow(list)
        
############################################################
#profits_se = pd.Series(profits)

plt.plot(alist,profits, alpha=.4)
plt.show()












##############################################
# test few data
symbols= convertible_bond_code 
#symbols= ['GOOG']  
#pnls1 = {i:dreader.DataReader(i,'yahoo','2019-01-01','2019-03-01') for i in symbols}

pnls2 = {i:ts.get_hist_data(i,start='2019-01-01',end='2019-03-01') for i in symbols}

# for modify
for i in symbols:        # 第二个实例
   pnls2[i]['close'].index = pnls2[i]['close'].index.astype('datetime64[ns]')