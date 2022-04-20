# encoding: utf-8
'''
用于计算RSRS各项指标

Created on 2020/08/19
@author: Hugo
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm  # 线性回归


class RSRS(object):

    '''
    计算各类RSRS
    ----------

    方法：
        1.get_RSRS 获取各类RSRS结果
            - df:index-date columns-必须包含close,high,low,pre_close,money
            - N:计算RSRS的参数
            - M:计算标准分的参数
            - method:计算RSRS的回归方法 ols或者wls

        2. cala_RSRS_z:计算RSRS标准分
        3. cala_revise_RSRS:计算RSRS修正标准分
        4. cala_negative_revise_RSRS:计算RSRS右偏修正标准分
        4. cala_passivation_RSRS:计算钝化RSRS
    '''

    def get_RSRS(self, df: pd.DataFrame, N: int, M: int, method: str) -> pd.DataFrame:
        '''
        计算各类RSRS

            df:index-date columns-|close|high|low|money|pre_close|
            N:计算RSRS
            M:修正标准分所需参数
            method:选择 ols 或 wls 回归
        '''
        selects = {'ols': (df, lambda x: self._cala_ols(x, 'low', 'high'), N),
                   'wls': (df, lambda x: self._cala_wls(x, 'low', 'high', 'money'), N)}

        returns = df['close'] / df['pre_close'] - 1  # 计算日度收益率
        ret_quantile = self._cala_ret_quantile(returns, N, M)  # 计算波动率百分位数

        rsrs_df = rolling_apply(*selects[method])  # 计算RSRS

        res_df = (rsrs_df.pipe(self.cala_RSRS_z, M)
                  .pipe(self.cala_revise_RSRS)
                  .pipe(self.cala_negative_revise_RSRS)
                  .pipe(self.cala_passivation_RSRS, ret_quantile))

        return res_df.drop(columns='R_2').iloc[M:]

    @staticmethod
    def cala_RSRS_z(df: pd.DataFrame, M: int) -> pd.DataFrame:
        '''
        标准分

            df:index-date columns-|RSRS|R_2|
        '''

        df['RSRS_z'] = (df['RSRS'] - df['RSRS'].rolling(M).mean()
                        ) / df['RSRS'].rolling(M).std()

        return df

    @staticmethod
    def cala_revise_RSRS(df: pd.DataFrame) -> pd.DataFrame:
        '''
        修正标准分

            df:index-date columns-|RSRS_z|R_2|
        '''

        df['RSRS_revise'] = df['RSRS_z'] * df['R_2']

        return df

    @staticmethod
    def cala_negative_revise_RSRS(df: pd.DataFrame) -> pd.DataFrame:
        '''
        右偏修正标准分RSRS
            df:index-date columns - |RSRS_revise|RSRS|
        '''
        df['RSRS_negative_r'] = df['RSRS_revise'] * df['RSRS']

        return df

    @staticmethod
    def cala_passivation_RSRS(df: pd.DataFrame, ret_quantile: pd.Series) -> pd.DataFrame:
        '''
        钝化RSRS
            df:index-date columns - |RSRS_z|R_2|
            ret_quantile:收益波动率百分位
        '''

        df['RSRS_passivation'] = df['RSRS_z'] * \
            np.power(df['R_2'], 2 * ret_quantile.reindex(df.index))

        return df

    # 计算wls回归的beta,r-squared

    @staticmethod
    def _cala_wls(df: pd.DataFrame, x_col: str, y_col: str, vol_col: str) -> pd.DataFrame:

        idx = df.index[-1]

        # 过滤NAN
        x = df[x_col].fillna(0)
        y = df[y_col].fillna(0)

        vol = df[vol_col]  # 成交量/成交额

        X = sm.add_constant(x)

        # 计算成交量权重
        def _get_vol_weights(slice_series: pd.Series) -> list:

            weights = slice_series / slice_series.sum()

            return weights.values.tolist()

        # 计算权重
        weights = _get_vol_weights(vol)

        results = sm.WLS(y, X, weights=weights).fit()

        # 计算beta
        BETA = results.params[1]
        # 计算r-squared
        Rsquared = results.rsquared

        return pd.DataFrame({'RSRS': BETA, 'R_2': Rsquared}, index=[idx])

    # 计算ols回归的beta,r-squared

    @staticmethod
    def _cala_ols(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:

        idx = df.index[-1]

        # 过滤NAN
        x = df[x_col].fillna(0)
        y = df[y_col].fillna(0)

        X = sm.add_constant(x)

        results = sm.OLS(y, X).fit()

        # 计算beta
        #BETA = np.linalg.lstsq(X, y, rcond=-1)[0][1]
        BETA = results.params[1]
        # 计算r-squared
        #Rsquared = np.corrcoef(x, y)[1, 0]**2
        Rsquared = results.rsquared

        return pd.DataFrame({'RSRS': BETA, 'R_2': Rsquared}, index=[idx])

    # 计算ret分位数
    @staticmethod
    def _cala_ret_quantile(ret: pd.Series, N: int, M: int) -> pd.Series:

        # 计算收益波动
        ret_std = ret.rolling(N, min_periods=1).apply(np.nanstd, raw=True)

        # 计算分位数
        ret_quantile = ret_std.rolling(M).apply(
            lambda x: x.rank(pct=True)[-1], raw=False)

        return ret_quantile


# 定义rolling_apply理论上应该比for循环快
# pandas.rolling.apply不支持多列
def rolling_apply(df, func, win_size) -> pd.Series:

    iidx = np.arange(len(df))

    shape = (iidx.size - win_size + 1, win_size)

    strides = (iidx.strides[0], iidx.strides[0])

    res = np.lib.stride_tricks.as_strided(
        iidx, shape=shape, strides=strides, writeable=True)

    # 这里注意func返回的需要为df或者ser
    return pd.concat((func(df.iloc[r]) for r in res), axis=0)  # concat可能会有点慢
