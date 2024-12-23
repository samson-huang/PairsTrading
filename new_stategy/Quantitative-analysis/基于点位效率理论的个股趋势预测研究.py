﻿# coding: utf-8
#import tushare as ts
import sys 
sys.path.append("C://Users//huangtuo//Documents//GitHub//PairsTrading//new_stategy//Quantitative-analysis//") 
import foundation_tushare 
import json


from Hugos_tools.Approximation import (Approximation, Mask_dir_peak_valley,
                                          Except_dir, Mask_status_peak_valley,
                                          Relative_values)

from Hugos_tools.performance import Strategy_performance
from collections import (defaultdict, namedtuple)
from typing import (List, Tuple, Dict, Union, Callable, Any)

import datetime as dt
import empyrical as ep
import numpy as np
import pandas as pd
import talib
import scipy.stats as st
from IPython.display import display

from sklearn.pipeline import Pipeline

from jqdatasdk import (auth, get_price, get_trade_days)

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 画图

def plot_pivots(peak_valley_df: pd.DataFrame,
                show_dir: Union[str,List,Tuple]='dir',
                show_hl: bool = True,
                show_point:bool = True,
                title: str = '',
                ax=None):

    if ax is None:

        fig, ax = plt.subplots(figsize=(18, 6))

        line = peak_valley_df.plot(y='close', alpha=0.6, title=title, ax=ax)

    else:

        line = peak_valley_df.plot(y='close', alpha=0.6, title=title, ax=ax)

    if show_hl:

        peak_valley_df.plot(ax=line,
                            y='PEAK',
                            marker='o',
                            color='r',
                            mec='black')

        peak_valley_df.plot(ax=line,
                            y='VALLEY',
                            marker='o',
                            color='g',
                            mec='black')
    
    if show_point:
        
        peak_valley_df.dropna(subset=['POINT']).plot(ax=line,
                                                     y='POINT',
                                                     color='darkgray',
                                                     ls='--')
    if show_dir:

        peak_valley_df.plot(ax=line,
                            y=show_dir,
                            secondary_y=True,
                            alpha=0.3,
                            ls='--')

    return line


def get_clf_wave(price: pd.DataFrame,
                 rate: float,
                 method: str,
                 except_dir: bool = True,
                 show_tmp: bool = False,
                 dropna: bool = True) -> pd.DataFrame:
    
    
    if except_dir:
        
        # 修正
        perpare_data = Pipeline([('approximation', Approximation(rate, method)),
                ('mask_dir_peak_valley',Mask_status_peak_valley('dir')),
                ('except', Except_dir('dir')),
                ('mask_status_peak_valley', Mask_dir_peak_valley('status'))
                ])
    else:
        
       # 普通
        perpare_data = Pipeline([('approximation', Approximation(rate, method)),
                ('mask_dir_peak_valley',Mask_dir_peak_valley('dir')),
                ('mask_status_peak_valley', Mask_status_peak_valley('dir'))])
        
   

    return perpare_data.fit_transform(price) 
    
    
# 使用ts


# 请根据自己的情况填写ts的token
setting = json.load(open('C:\config\config.json'))
my_ts  = foundation_tushare.TuShare(setting['token'], max_retry=60)


start = '20050101'
end = '20220415'
index_df = my_ts.query('index_daily', ts_code='000300.SH', 
start_date=start, end_date=end,fields='trade_date,close,high,low,open')    
hs300=index_df
hs300.index = pd.to_datetime(hs300.trade_date)
del hs300['trade_date']
hs300.sort_index(inplace=True)  # 排序

#hs300.to_csv('hs300.csv')

#针对MACD价格划分方式的改进     

# 方式1:划分上下行

flag_frame1: pd.DataFrame = get_clf_wave(hs300,None,'a',False)

begin, end = '2020-02-01','2020-07-20'
flag_df1 = flag_frame1.loc[begin:end,['close','dir']]

flag_df1 = flag_df1.rename(columns={'dir':'方式1划分上下行'})
	
	
line = flag_frame1.loc['2021-01-01':'2021-07-30'].plot(figsize=(18, 6), y='close', color='red',
                    title='沪深300收盘价、DIF线与DEA线(2021-01-04至2021-07-30)')

flag_frame1.loc['2021-01-01':'2021-07-30'].plot(ax=line, y=['dif', 'dea'],
             secondary_y=True, color=['#3D89BE', 'darkgray']);


#划分方式1

# 画图
line = flag_df1.plot(y='方式1划分上下行', secondary_y=True, figsize=(
    18, 5), ls='--', color='darkgray', title='沪深300与上下行划分(方式1)')

flag_df1.plot(y='close', ax=line, color='r');

#划分方式二
# 方式2:划分上下行
flag_frame2: pd.DataFrame = get_clf_wave(hs300,0.5,'b',False)

flag_df2 = flag_frame2.loc[begin:end,['close','dir']]
flag_df2 = flag_df2.rename(columns={'dir':'方式2划分上下行'})


# 画图
line = flag_df2.plot(y='方式2划分上下行', secondary_y=True, figsize=(
    18, 5), ls='--', color='darkgray', title='沪深300与上下行划分(方式2,Rate=0.5)')

flag_df2.plot(y='close', ax=line, color='r');


#划分方式三
# 方式3:划分上下行

flag_frame3: pd.DataFrame = get_clf_wave(hs300,2,'c',False)

flag_df3 = flag_frame3.loc[begin:end,['close','dir']]
flag_df3 = flag_df3.rename(columns={'dir':'方式3划分上下行'})
	
# 画图
line = flag_df3.plot(y='方式3划分上下行', secondary_y=True, figsize=(
    18, 5), ls='--', color='darkgray', title='沪深300与上下行划分(方式3,Rate=2)')

flag_df3.plot(y='close', ax=line, color='r');    


#高低点识别与异常端点
# 方式3:划分上下行-修正与普通的划分方式对比

status_frame: pd.DataFrame = get_clf_wave(hs300, 2, 'c', True)
dir_frame: pd.DataFrame = get_clf_wave(hs300, 2, 'c', False)
	
fig, axes = plt.subplots(2,figsize=(18,12))
# 画图
plot_pivots(dir_frame.iloc[330:450],
            show_dir=['dir'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3,Rate=2)',ax=axes[0])

plot_pivots(status_frame.iloc[330:450],
            show_dir=['status'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3-修正,Rate=2)',ax=axes[1]);	
            

fig, axes = plt.subplots(2,figsize=(18,12))
# 画图
plot_pivots(dir_frame.iloc[:50],
            show_dir=['dir'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3,Rate=2)',ax=axes[0])

plot_pivots(status_frame.iloc[:50],
            show_dir=['status'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3-修正,Rate=2)',ax=axes[1]);


fig, axes = plt.subplots(2,figsize=(18,12))
# 画图
plot_pivots(dir_frame.loc['2019-01-01':'2021-07-30'],
            show_dir=['dir'],
            show_hl=True,
            title='沪深300指数波段划分结果展示（2019-01-03 至 2021-07-30）(方法3,Rate=2)',ax=axes[0])

plot_pivots(status_frame.loc['2019-01-01':'2021-07-30'],
            show_dir=['status'],
            show_hl=True,
            title='沪深300指数波段划分结果展示（2019-01-03 至 2021-07-30）(方法3-修正,Rate=2)',ax=axes[1]);                     
            
fig, axes = plt.subplots(2,figsize=(18,12))
# 画图
plot_pivots(dir_frame.loc['2021-05-01':'2021-09-15'],
            show_dir=['dir'],
            show_hl=True,
            title='当前点位与它的前两个端点(方法3,Rate=2)',ax=axes[0])

plot_pivots(status_frame.loc['2021-05-01':'2021-09-15'],
            show_dir=['status'],
            show_hl=True,
            title='当前点位与它的前两个端点(方法3-修正,Rate=2)',ax=axes[1]); 
                       
#上下行划分分析
fig, axes = plt.subplots(2,figsize=(18,12))

plot_pivots(status_frame,
            show_dir=False,
            show_hl=False,
            title='沪深300指数波段划分结果展示(方法3-修正,Rate=2)',ax=axes[0])

plot_pivots(dir_frame,
            show_dir=False,
            show_hl=False,
            title='沪深300指数波段划分结果展示(方法3,Rate=2)',ax=axes[1])
'''
基于日度沪深300指数的波段划分结上图 。纵观整个时间区间，无论是大趋势还是长期震荡，波段的划分几乎与价格走势重合，这也是我们所期望的结果。在考察范围内的3890个(2005-09-02至2021-08-31)交易日中：
修正方案下
共有216个波段，其中108个上涨波段，108个下降波段
普通方案下
共有113个波段,其中57个上涨波段，56个下降波段
'''
class Segment_stats():
    '''
    分析上下行划分的统计数据
    '''
    def __init__(self,
                 frame: pd.DataFrame,
                 flag_col: str,
                 drop: bool = True) -> None:

        self.data = frame
        self.flag_col = flag_col
        self.drop = drop
        self._COL_NAME_MAP = {-1: '下跌波段', 1: '上涨波段'}
        self._prepare()

    def winsorize_std(self) -> pd.DataFrame:
        '''标准差去极值'''

        ret = self.stats_frame.groupby('g')['log_ret'].sum()
        up = ret.mean() + ret.std() * 2
        low =  ret.mean() - ret.std() * 2

        group_id = ret[(ret <= up) & (ret >= low)].index.tolist()
        return self.stats_frame.query('g == @group_id')

    def _prepare(self) -> None:
        '''预处理'''
        stats_df = self.data[['close', self.flag_col]].copy()

        stats_df['log_ret'] = np.log(stats_df['close'] /
                                     stats_df['close'].shift(1))

        if self.drop:
            stats_df = stats_df.dropna(subset=[self.flag_col])

        stats_df['g'] = (stats_df[self.flag_col] !=
                         stats_df[self.flag_col].shift(1)).cumsum()

        self.stats_frame = stats_df

    def stats_summary(self, winsorize: bool = False) -> pd.DataFrame:
        '''简易统计报告'''

        if winsorize:

            stats_frame = self.winsorize_std()

        else:

            stats_frame = self.stats_frame

        SUMMARY_STATS = [
            ('均值', lambda x: x.mean()), ('标准差', lambda x: x.std()),
            ('最大值', lambda x: x.max()), ('最小值', lambda x: x.min()),
            ('样本数量', lambda x: len(x))
        ]

        group_stats = stats_frame.groupby([self.flag_col,
                                           'g'])['log_ret'].sum()

        stats = group_stats.groupby(level=self.flag_col).agg(SUMMARY_STATS)

        stats.index = stats.index.map(self._COL_NAME_MAP)

        return display(
            stats.style.format('{:.2%}', subset=['均值', '标准差', '最大值', '最小值']))

    def ttest_segment(self, winsorize: bool = False) -> pd.DataFrame:
        '''波段的T检验'''

        if winsorize:

            stats_frame = self.winsorize_std()

        else:

            stats_frame = self.stats_frame

        t_test = stats_frame.groupby(
            self.flag_col)['log_ret'].apply(lambda x: pd.Series(
                st.ttest_1samp(x.dropna(), 0), index=['statistic', 'pvalue']))

        t_test = t_test.unstack()
        t_test.index = t_test.index.map(self._COL_NAME_MAP)

        return display(
            t_test.style.format({
                'statistic': '{:.2f}',
                'pvalue': '{:.4f}'
            }))

    def plot_segment_ret_hist(self, winsorize: bool = False, **kw):

        if winsorize:

            stats_frame = self.winsorize_std()

        else:

            stats_frame = self.stats_frame

        group_ret = stats_frame.groupby([self.flag_col, 'g'])['log_ret'].sum()

        ret_min = group_ret.min()
        ret_max = group_ret.max()
        
        labels = ['<=5%', '6%-10%', '11%-15%', '16%-20%', '>20%']
        bins = [ret_min, 0.05, 0.1, 0.15, 0.2, ret_max]
        
        if ret_max <= 0.2:
            bins = bins[:-1]
            labels = labels[:-1]
            labels[-1] = '16%-{:.0%}'.format(ret_max)
            
        if ret_min >= 0.05:
            bins = bins[1:]
            labels = labels[1:]
            labels[0] = '6%-{:.0%}'.format(ret_min)
            
        ser: pd.Series = pd.cut(
            group_ret,
            bins=bins,
            labels=labels)

        df = ser.groupby(self.flag_col).value_counts().unstack(level=0)
        df.rename(columns=self._COL_NAME_MAP, inplace=True)

        return df.plot.bar(**kw)            

#修正后波段划分情况

stats_summary = Segment_stats(status_frame,'status')

print('未去极值波段划分情况')
stats_summary.stats_summary()
stats_summary.ttest_segment()


print('去极值波段划分情况')
stats_summary.stats_summary(True)

stats_summary.ttest_segment(True)

fig,axes = plt.subplots(1,2,figsize=(18,6))

stats_summary.plot_segment_ret_hist(title='未去极值',ax=axes[0])
stats_summary.plot_segment_ret_hist(winsorize=True,title='去极值',ax=axes[1]);


#未修正的波段划分情况
stats_summary = Segment_stats(dir_frame,'dir')

print('未去极值波段划分情况')
stats_summary.stats_summary()
stats_summary.ttest_segment()


print('去极值波段划分情况')
stats_summary.stats_summary(True)

stats_summary.ttest_segment(True)

fig,axes = plt.subplots(1,2,figsize=(18,6))

stats_summary.plot_segment_ret_hist(title='未去极值',ax=axes[0])
stats_summary.plot_segment_ret_hist(winsorize=True,title='去极值',ax=axes[1]);



#点位效率的理论介绍与计算

rv = Relative_values('dir')
rv_df:pd.DataFrame = rv.fit_transform(dir_frame)

test_rv_df:pd.DataFrame = rv_df[['close','relative_time','relative_price']].copy()
for i in range(1,25):

    test_rv_df[i] = test_rv_df['close'].pct_change(i).shift(-i)
    
drop_tmp = test_rv_df.dropna(subset=['relative_price'])



drop_tmp[['close', 'relative_price', 'relative_time']].plot(figsize=(18, 12),
                                                            subplots=True);
                                                            



#应用部分
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin


X = drop_tmp[['relative_price','relative_time']].values

n_clusters = 3

k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
k_means.fit(X)

k_means_cluster_centers = k_means.cluster_centers_  # 获取聚类核心点

k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)  # 计算一个点和一组点之间的最小距离,默认欧式距离


k_mean_cluster_frame:pd.DataFrame = drop_tmp.copy()

k_mean_cluster_frame['label'] = k_means_labels




import warnings
from numbers import Number
from functools import partial
import math
from seaborn.palettes import color_palette
from seaborn.distributions import _DistributionPlotter
from seaborn._statistics import KDE

#    
def plot_bivariate_density(
    self,
    common_norm,
    fill,
    levels,
    thresh,
    color,
    legend,
    cbar,
    cbar_ax,
    cbar_kws,
    estimate_kws,
    **contour_kws,
):

    contour_kws = contour_kws.copy()

    estimator = KDE(**estimate_kws)

    if not set(self.variables) - {"x", "y"}:
        common_norm = False

    all_data = self.plot_data.dropna()

    # Loop through the subsets and estimate the KDEs
    densities, supports = {}, {}

    for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):

        # Extract the data points from this sub set and remove nulls
        sub_data = sub_data.dropna()
        observations = sub_data[["x", "y"]]

        # Extract the weights for this subset of observations
        if "weights" in self.variables:
            weights = sub_data["weights"]
        else:
            weights = None

        # Check that KDE will not error out
        variance = observations[["x", "y"]].var()
        if any(math.isclose(x, 0)
                for x in variance) or variance.isna().any():
            msg = "Dataset has 0 variance; skipping density estimate."
            warnings.warn(msg, UserWarning)
            continue

        # Estimate the density of observations at this level
        observations = observations["x"], observations["y"]
        density, support = estimator(*observations, weights=weights)

        # Transform the support grid back to the original scale
        xx, yy = support
        if self._log_scaled("x"):
            xx = np.power(10, xx)
        if self._log_scaled("y"):
            yy = np.power(10, yy)
        support = xx, yy

        # Apply a scaling factor so that the integral over all subsets is 1
        if common_norm:
            density *= len(sub_data) / len(all_data)

        key = tuple(sub_vars.items())
        densities[key] = density
        supports[key] = support

    # Define a grid of iso-proportion levels
    if thresh is None:
        thresh = 0
    if isinstance(levels, Number):
        levels = np.linspace(thresh, 1, levels)
    else:
        if min(levels) < 0 or max(levels) > 1:
            raise ValueError("levels must be in [0, 1]")

    # Transform from iso-proportions to iso-densities
    if common_norm:
        common_levels = self._quantile_to_level(
            list(densities.values()),
            levels,
        )
        draw_levels = {k: common_levels for k in densities}
    else:
        draw_levels = {
            k: self._quantile_to_level(d, levels)
            for k, d in densities.items()
        }

    # Get a default single color from the attribute cycle
    if self.ax is None:
        default_color = "C0" if color is None else color
    else:
        scout, = self.ax.plot([], color=color)
        default_color = scout.get_color()
        scout.remove()

    # Define the coloring of the contours
    if "hue" in self.variables:
        for param in ["cmap", "colors"]:
            if param in contour_kws:
                msg = f"{param} parameter ignored when using hue mapping."
                warnings.warn(msg, UserWarning)
                contour_kws.pop(param)
    else:

        # Work out a default coloring of the contours
        coloring_given = set(contour_kws) & {"cmap", "colors"}
        if fill and not coloring_given:
            cmap = self._cmap_from_color(default_color)
            contour_kws["cmap"] = cmap
        if not fill and not coloring_given:
            contour_kws["colors"] = [default_color]

        # Use our internal colormap lookup
        cmap = contour_kws.pop("cmap", None)
        if isinstance(cmap, str):
            cmap = color_palette(cmap, as_cmap=True)
        if cmap is not None:
            contour_kws["cmap"] = cmap

    # Loop through the subsets again and plot the data
    for sub_vars, _ in self.iter_data("hue"):

        if "hue" in sub_vars:
            color = self._hue_map(sub_vars["hue"])
            if fill:
                contour_kws["cmap"] = self._cmap_from_color(color)
            else:
                contour_kws["colors"] = [color]

        ax = self._get_axes(sub_vars)

        # Choose the function to plot with
        # TODO could add a pcolormesh based option as well
        # Which would look something like element="raster"
        if fill:
            contour_func = ax.contourf
        else:
            contour_func = ax.contour

        key = tuple(sub_vars.items())
        if key not in densities:
            continue
        density = densities[key]
        xx, yy = supports[key]

        label = contour_kws.pop("label", None)

        cset = contour_func(
            xx,
            yy,
            density,
            levels=draw_levels[key],
            **contour_kws,
        )
        
        ax.clabel(cset,inline=True) # 就加了一个这个
        if "hue" not in self.variables:
            cset.collections[0].set_label(label)

        # Add a color bar representing the contour heights
        # Note: this shows iso densities, not iso proportions
        # See more notes in histplot about how this could be improved
        if cbar:
            cbar_kws = {} if cbar_kws is None else cbar_kws
            ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)

    # --- Finalize the plot
    ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
    self._add_axis_labels(ax)

    if "hue" in self.variables and legend:

        # TODO if possible, I would like to move the contour
        # intensity information into the legend too and label the
        # iso proportions rather than the raw density values

        artist_kws = {}
        if fill:
            artist = partial(mpl.patches.Patch)
        else:
            artist = partial(mpl.lines.Line2D, [], [])

        ax_obj = self.ax if self.ax is not None else self.facets
        self._add_legend(
            ax_obj,
            artist,
            fill,
            False,
            "layer",
            1,
            artist_kws,
            {},
        )
        
_DistributionPlotter.plot_bivariate_density = plot_bivariate_density

def plot_simple_cluster(k_mean_cluster_frame:pd.DataFrame,k_means_cluster_centers:np.array,x:str,y:str,hue:str):
    '''画聚类图
    
    k_means_cluster_centers:为质心
    '''
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#4EACC5', '#FF9C34', '#4E9A06']

    scatter = sns.scatterplot(data=k_mean_cluster_frame,
                            x=x,
                            y=y,
                            hue=hue,
                            ax=ax,
                            palette=colors)
    
    for i, (r, c) in enumerate(k_means_cluster_centers):

        scatter.plot(r,
                    c,
                    'o',
                    markerfacecolor=colors[i],
                    markeredgecolor='k',
                    markersize=8)

    return scatter
    

plot_simple_cluster(k_mean_cluster_frame,k_means_cluster_centers,x='relative_price',y='relative_time',hue='label');


mel_df = pd.melt(k_mean_cluster_frame,id_vars=['label'],value_vars=list(range(1,25)),var_name=['day'])
slice_df = mel_df.query('label==0').dropna() 
slice_df['day'] = slice_df['day'].astype(int)
fig,ax = plt.subplots(figsize=(14,9))
sns.kdeplot(data=slice_df, x='day',y='value',cbar=True,cmap="coolwarm");


#朴素贝叶斯与逻辑回归
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib # 用于模型导出

train_df = test_rv_df.loc[:'2022-01-01'].dropna()

test_df = test_rv_df.loc['2022-01-01':]

x_test = test_df[['relative_time','relative_price']]

tscv = TimeSeriesSplit(n_splits=5,max_train_size=180)

nb = GaussianNB()

lr = LogisticRegression()
for i,(train_index, test_index) in enumerate(tscv.split(train_df)):

    x_train = train_df.iloc[train_index][['relative_time','relative_price']]
    y_train = train_df.iloc[train_index][1]
    y_sign = np.where(y_train > 0.,1,0)
    lr.fit(x_train,y_sign)
    nb.fit(x_train,y_sign)


df = pd.DataFrame()
next_ret = test_rv_df['close'].pct_change().shift(-1)
next_ret1 = test_rv_df['close'].pct_change()
df['GaussianNB'] = next_ret.loc[test_df.index] * nb.predict(x_test)
df['LogisticRegression'] = next_ret.loc[test_df.index] * lr.predict(x_test)
ep.cum_returns(df).plot(figsize=(18,6))

ep.cum_returns(next_ret1.loc[x_test.index]).plot(color='darkgray',label='HS300')
plt.legend();  

###########复用评测#################

    def summary(back_testing):

        back_df = back_testing

        index_name = '年化收益率,累计收益率,夏普比率,最大回撤,持仓总天数,交易次数,平均持仓天数,获利天数, \
        亏损天数,胜率(按天),平均盈利率(按天),平均亏损率(按天),平均盈亏比(按天),盈利次数,亏损次数, \
        单次最大盈利,单次最大亏损,胜率(按此),平均盈利率(按次),平均亏损率(按次),平均盈亏比(按次)'.split(
            ',')

        # 寻找标列
        mark_list = [x for x in back_df.columns if x.split('_')[-1] == 'MARK']

        temp = []
        mark_size = len(mark_list)  # 列数

        if mark_size > 1:

            for m in mark_list:

                df = pd.DataFrame(risk_indicator(
                    back_df, m), index=index_name)
                temp.append(df)

            return pd.concat(temp, axis=1)

        else:

            return pd.DataFrame(risk_indicator(back_df, m), index=index_name)

    # 计算风险指标

    def risk_indicator(x_df, mark_col):
        '''
        传入经back_testing

        '''
        df = x_df.copy()

        summary_dic = {}

        # 格式化数据
        def format_x(x):
            return '{:.2%}'.format(x)

        # 获取回测数据
        df['pct_chg'] = df['pct_chg']/100
        df['NEXT_RET'] = df['pct_chg'].shift(-1)

        NOT_NAN_RET = df['NEXT_RET'].dropna()*df[mark_col]
        RET = df['NEXT_RET']*df[mark_col]

        CUM_RET = (1+RET).cumprod()  # series

        # 计算年化收益率
        annual_ret = CUM_RET.dropna()[-1]**(250/len(NOT_NAN_RET)) - 1

        # 计算累计收益率
        cum_ret_rate = CUM_RET.dropna()[-1] - 1

        # 最大回撤
        max_nv = np.maximum.accumulate(np.nan_to_num(CUM_RET))
        mdd = -np.min(CUM_RET / max_nv - 1)

        # 夏普
        sharpe_ratio = np.mean(NOT_NAN_RET) / \
            np.nanstd(NOT_NAN_RET, ddof=1)*np.sqrt(250)

        # 盈利次数
        temp_df = df.copy()

        diff = temp_df[mark_col] != temp_df[mark_col].shift(1)
        temp_df[mark_col+'_diff'] = diff.cumsum()
        cond = temp_df[mark_col] == 1
        # 每次开仓的收益率情况
        temp_df = temp_df[cond].groupby(mark_col+'_diff')['NEXT_RET'].sum()

        # 标记买入卖出时点
        mark = df[mark_col]
        pre_mark = np.nan_to_num(df[mark_col].shift(-1))
        # 买入时点
        trade = (mark == 1) & (pre_mark < mark)

        # 交易次数
        trade_count = len(temp_df)

        # 持仓总天数
        total = np.sum(mark)

        # 平均持仓天数
        mean_hold = total/trade_count
        # 获利天数
        win = np.sum(np.where(RET > 0, 1, 0))
        # 亏损天数
        lose = np.sum(np.where(RET < 0, 1, 0))
        # 胜率
        win_ratio = win/total
        # 平均盈利率（天）
        mean_win_ratio = np.sum(np.where(RET > 0, RET, 0))/win
        # 平均亏损率（天）
        mean_lose_ratio = np.sum(np.where(RET < 0, RET, 0))/lose
        # 盈亏比(天)
        win_lose = win/lose

        # 盈利次数
        win_count = np.sum(np.where(temp_df > 0, 1, 0))
        # 亏损次数
        lose_count = np.sum(np.where(temp_df < 0, 1, 0))
        # 单次最大盈利
        max_win = np.max(temp_df)
        # 单次最大亏损
        max_lose = np.min(temp_df)
        # 胜率
        win_rat = win_count/len(temp_df)
        # 平均盈利率（次）
        mean_win = np.sum(np.where(temp_df > 0, temp_df, 0))/len(temp_df)
        # 平均亏损率（天）
        mean_lose = np.sum(np.where(temp_df < 0, temp_df, 0))/len(temp_df)
        # 盈亏比(次)
        mean_wine_lose = win_count/lose_count

        summary_dic[mark_col] = [format_x(annual_ret), format_x(cum_ret_rate), sharpe_ratio, format_x(
            mdd), total, trade_count, mean_hold, win, lose, format_x(win_ratio), format_x(mean_win_ratio),
            format_x(mean_lose_ratio), win_lose, win_count, lose_count, format_x(
                max_win), format_x(max_lose),
            format_x(win_rat), format_x(mean_win), format_x(mean_lose), mean_wine_lose]

        return summary_dic

##############################

df1 = pd.DataFrame()
test1=test_rv_df['close']*0+1
df1['GaussianNB'] = test1.loc[test_df.index] * nb.predict(x_test)
df1['LogisticRegression'] = test1.loc[test_df.index] * lr.predict(x_test)
df1.columns=['GaussianNB_MARK','LogisticRegression_MARK']

test123=next_ret1*100
test123.columns = ['pct_chg'] 
test4=pd.merge(test123,df1,how='inner', left_index=True, right_index=True)
test4.columns =['pct_chg','GaussianNB_MARK','LogisticRegression_MARK']

test4=pd.merge(test4,df,how='inner', left_index=True, right_index=True)
summary(test4)


                                                            
                                                            


            