import tushare as ts
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

ts.get_sz50s()

sz50s=ts.get_sz50s()

#convert np.array
sz50s_code=sz50s["code"].values
convertible_bond_code=(['300059','123006'])



####################从wind取数据#######################################
from WindPy import *
w.start()
wsddata1=w.wsd('123006.sz', "open,high,low,close,volume,amt",'20190101','20190301', "Fill=Previous")
wsddata1.Data
#################################################################


# test few data
symbols= convertible_bond_code 
#symbols= ['GOOG']  
#pnls1 = {i:dreader.DataReader(i,'yahoo','2019-01-01','2019-03-01') for i in symbols}

pnls2 = {i:ts.get_hist_data(i,start='2019-01-01',end='2019-03-01') for i in symbols}

# for modify
for i in symbols:        # 第二个实例
   pnls2[i]['close'].index = pnls2[i]['close'].index.astype('datetime64[ns]')