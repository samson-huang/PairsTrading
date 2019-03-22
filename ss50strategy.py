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

sz50s_code_test=sz50s_code[0:4]
	
# test few data
symbols= sz50s_code_test 
#symbols= ['GOOG']  
#pnls1 = {i:dreader.DataReader(i,'yahoo','2019-01-01','2019-03-01') for i in symbols}

pnls2 = {i:ts.get_hist_data(i,start='2019-01-01',end='2019-03-01') for i in symbols}
	
	
#modify index type
#pnls['600000']['close'].index = pnls['600000']['close'].index.astype('datetime64[ns]')

# for modify
for i in symbols:        # 第二个实例
   pnls2[i]['close'].index = pnls2[i]['close'].index.astype('datetime64[ns]')



###########################plot########################
# solve  chinese dislay
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plot them
plt.plot(pnls2['600000']['close'], label='浦发银行')
plt.plot(pnls2['600016']['close'], label='民生银行')
plt.plot(pnls2['600019']['close'], label='宝钢股份')
plt.plot(pnls2['600028']['close'], label='中国石化')
	
	
# generate a legend box
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
       ncol=4, mode="expand", borderaxespad=0.)
 
# annotate an important value
#plt.annotate("Important value", (55,20), xycoords='data',
#         xytext=(5, 38),
#         arrowprops=dict(arrowstyle='->'))
plt.show()





##############difference#############
def total_data(data_p):
    # for modify
    symbols_func=sz50s["code"].values[1:4]
    total_data =pd.DataFrame([data_p['600000']['close'].sort_index().pct_change()])
    for i in symbols_func:        # 第二个实例
       total_data= total_data.append(pd.DataFrame([data_p[i]['close'].sort_index().pct_change()]))
    return(total_data)

################################################
total_data=total_data(pnls2)


total_data.index=sz50s["code"].values[0:4]
total_data=total_data.dropna(axis=1,how='all') 

plt.plot(total_data.T, alpha=.4);
plt.xlabel('time')
plt.ylabel('returns')
##################assign#################################

def assign_portfolio(returns,assign_weights):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(assign_weights.T)
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    #if sigma > 2:
        #return random_portfolio(returns)
    return mu, sigma
#########################################################
############################################################
def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    #if sigma > 2:
        #return random_portfolio(returns)
    return mu, sigma

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)
#################################################################
n_portfolios = 10000
means, stds = np.column_stack([
    random_portfolio(total_data) 
    for _ in range(n_portfolios)
])


plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')



############################################################
#######################optimal portfolio function##########################################
def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

#############################################################
weights, returns, risks = optimal_portfolio(total_data)

###########annotate optimal weights####################
optimal_mu,optimal_sigma=assign_portfolio(total_data,weights)
show_max=str(optimal_mu)+' '+str(optimal_sigma)


plt.plot(stds, means, 'o')


plt.annotate(show_max,xytext=(optimal_sigma,optimal_mu),xy=(optimal_sigma,optimal_mu))



plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')
plt.plot(optimal_sigma,optimal_mu,'ks')
plt.show()
##print (weights)



##########################
#########################
#########################
cons = ts.get_apis()
df_day =ts.bar("IF1801",conn=cons,asset='X',freq='D')