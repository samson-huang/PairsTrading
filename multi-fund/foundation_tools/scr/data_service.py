"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-01-12 16:58:39
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-01-12 17:00:39
Description:
"""
from typing import List, Tuple, Union
import pandas as pd
from knight_scr import db_model
from knight_scr.scr.db_tool import DataMySqlLayer
from knight_scr.scr.sql_service import Tdaysoffset, get_ind_classify, get_ind_price, get_trade_days, get_ts_etf_price, get_ts_index_price
from knight_scr.scr.utils import add_uuid

def get_stock_pool() -> pd.DataFrame:
    """股票池"""
    dml = DataMySqlLayer('mysql+mysqlconnector://root:dacc54e8757706c5@101.132.70.214:3306/100score')
    dml.connect()
    session = dml.Session()
    model = dml.auto_db_base.classes['stock_pools']
    q = session.query(model).filter(model.focus_user_ids.like('c3c2c15c-2675-447a-9578-1a053fe12a01'), model.type == 1, model.status == 1)
    return pd.read_sql(q.statement, dml.engine)


def get_daily_price(start: str, end: str, fields: Union[(str, List)], codes: Union[(str, List)]=None) -> pd.DataFrame:
    turnover_filter = [
     db_model.Daily_basic.trade_date >= start,
     db_model.Daily_basic.trade_date <= end]
    daily_filter = [
     db_model.Daily.trade_date >= start,
     db_model.Daily.trade_date <= end]
    adj_factor_filter = [
     db_model.Adj_factor.trade_date >= start,
     db_model.Adj_factor.trade_date <= end]
    if codes is not None:
        if isinstance(codes, str):
            codes = [
             codes]
        else:
            turnover_filter.append(db_model.Daily_basic.code.in_(codes))
            daily_filter.append(db_model.Daily.code.in_(codes))
            adj_factor_filter.append(db_model.Adj_factor.code.in_(codes))
    dml = DataMySqlLayer()
    dml.connect()
    session = dml.Session()
    if isinstance(fields, str):
        fields = [
         fields]
    have_turnover_field = False
    if 'turnover_rate_f' in fields:
        turnover_fields = tuple((getattr(db_model.Daily_basic, i) for i in ))
        turnover_q = ((session.query)(*turnover_fields).filter)(*turnover_filter)
        turnover = pd.read_sql(turnover_q.statement, dml.engine)
        COND3 = turnover['code'].str.endswith('BJ')
        turnover = turnover[~COND3]
        pivot_turnover = pd.pivot_table(turnover,
          index='trade_date', columns='code', values=['turnover_rate_f'])
        fields.remove('turnover_rate_f')
        have_turnover_field = True
    daily_fields = tuple((getattr(db_model.Daily, i) for i in ))
    adj_fields = tuple((getattr(db_model.Adj_factor, i) for i in ))
    daily_q = ((session.query)(*daily_fields).filter)(*daily_filter)
    adj_factor_q = ((session.query)(*adj_fields).filter)(*adj_factor_filter)
    price = pd.read_sql(daily_q.statement, dml.engine)
    adj_factor = pd.read_sql(adj_factor_q.statement, dml.engine)
    COND1 = price['code'].str.endswith('BJ')
    COND2 = adj_factor['code'].str.endswith('BJ')
    price = price[~COND1]
    adj_factor = adj_factor[~COND2]
    pivot_price = pd.pivot_table(price,
      index='trade_date', columns='code', values=fields)
    pivot_adj = pd.pivot_table(adj_factor,
      index='trade_date', columns='code', values='adj_factor')
    for field in fields:
        if field not in ('vol', 'amount'):
            df_, adj_ = pivot_price[field].align(pivot_adj, join='left')
            pivot_price[field] = df_ * adj_
        if have_turnover_field:
            pivot_price = pd.concat((pivot_price, pivot_turnover), axis=1)
        return pivot_price