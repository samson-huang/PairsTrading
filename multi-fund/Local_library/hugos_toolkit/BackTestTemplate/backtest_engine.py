"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2022-10-27 20:34:02
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2022-11-22 16:09:44
Description: 回测所需配件
"""
import datetime
from collections import namedtuple
from typing import Dict

import backtrader as bt
import backtrader.feeds as btfeeds
import numpy as np
import pandas as pd

from .bt_strategy import SignalStrategy

class OrderAnalyzer(bt.analyzers.Analyzer):
    def __init__(self):
        self.orders = []

    def notify_order(self, order):
        # 记录订单信息
        order_info = {
            'ref': order.ref,  # 订单参考编号
            'status': order.getstatusname(),  # 订单状态
            'size': order.size,  # 订单数量
            'price': order.price,  # 订单价格
            'value': order.executed.value,  # 订单金额
            'reason': order.info.reason,  # 订单原因
            'date': bt.num2date(order.data.datetime[0]),  # 订单日期
            'data': order.data._name,  # 相关数据源名称
            'type': 'Buy' if order.isbuy() else 'Sell',  # 订单类型
        }
        self.orders.append(order_info)

    def get_analysis(self):
        return self.orders

class TradeAnalyzer_1(bt.analyzers.Analyzer):
    def __init__(self):
        self.trades = []
        self.cumprofit = 0.0

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.cumprofit += trade.pnl

        # 记录交易信息
        trade_info = {
            'ref': trade.ref,  # 交易参考编号
            'buy_date': bt.num2date(trade.dtopen),  # 买入日期
            'buy_price': trade.price,  # 买入价格
            'buy_size': trade.size,  # 买入数量
            'sell_date': bt.num2date(trade.dtclose),  # 卖出日期
            'sell_price': trade.priceclosed,  # 卖出价格
            'sell_size': trade.sizeclosed,  # 卖出数量
            'pnl': trade.pnl,  # 盈亏金额
        }
        self.trades.append(trade_info)

    def get_analysis(self):
        return {
            'trades': self.trades,
            'cumprofit': self.cumprofit,
        }
##############################################################
class TradeLogger(bt.Analyzer):
    def __init__(self):
        self.trades = []

    def notify_trade(self, trade):
        if not trade:
            return

        #if not trade.isclosed:
            #return
        # 当交易发生时调用
        self.trades.append({
            'ref': trade.ref,  # 交易参考编号
            'buy_date': bt.num2date(trade.dtopen),  # 买入日期
            'buy_price': trade.price,  # 买入价格
            'buy_size': trade.size,  # 买入数量
            #'sell_date': bt.num2date(trade.dtclose),  # 卖出日期
            #'sell_price': trade.priceclosed,  # 卖出价格
            #'sell_size': trade.sizeclosed,  # 卖出数量
            'pnl': trade.pnl,  # 盈亏金额
        })

    def get_analysis(self):
        return self.trades

##########我自己添加的分析函数###
class TradeListAnalyzer(bt.Analyzer):
    """
    交易列表分析器
    https://community.backtrader.com/topic/1274/closed-trade-list-including-mfe-mae-analyzer/2
    """

    def __init__(self):
        self.trades = []
        self.cum_profit = 0.0

    def get_analysis(self) -> tuple:
        """
        获取分析数据
        @return: 交易订单列表，交易日期
        """
        trade_list_df = pd.DataFrame(self.trades)
        return trade_list_df, self._get_trade_date(trade_list_df)
        #return trade_list_df

    def _get_trade_date(self, trade_list_df):
        """
        获取交易日期
        @return: 交易日期，获取某只股票的买卖日期，
        返回字典，key为股票名，value为(买入日期列表，卖出日期列表)
        """
        trade_dict = dict()
        if not trade_list_df.empty:
            # 分组，找出买卖日期
            grouped = trade_list_df.groupby('股票')
            for name, group in grouped:
                buy_date_list = list(group['买入日期'])
                sell_date_list = list(group['卖出日期'])
                # 判断是否有买卖日期
                if trade_dict.get(name) is None:
                    trade_dict[name] = (buy_date_list, sell_date_list)
                else:
                    trade_dict[name][0].extend(buy_date_list)
                    trade_dict[name][1].extend(sell_date_list)
        return trade_dict

    def notify_trade(self, trade):
        if trade.isclosed:

            total_value = self.strategy.broker.getvalue()

            dir = 'short'
            if trade.history[0].event.size > 0: dir = 'long'

            pricein = trade.history[len(trade.history) - 1].status.price
            priceout = trade.history[len(trade.history) - 1].event.price
            datein = bt.num2date(trade.history[0].status.dt)
            dateout = bt.num2date(trade.history[len(trade.history) - 1].status.dt)
            if trade.data._timeframe >= bt.TimeFrame.Days:
                datein = datein.date()
                dateout = dateout.date()

            pcntchange = 100 * priceout / pricein - 100
            pnl = trade.history[len(trade.history) - 1].status.pnlcomm
            pnlpcnt = 100 * pnl / total_value
            barlen = trade.history[len(trade.history) - 1].status.barlen
            pbar = pnl / barlen
            self.cum_profit += pnl

            size = value = 0.0
            for record in trade.history:
                if abs(size) < abs(record.status.size):
                    size = record.status.size
                    value = record.status.value

            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen + 1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen + 1))
            hp = 100 * (highest_in_trade - pricein) / pricein
            lp = 100 * (lowest_in_trade - pricein) / pricein
            if dir == 'long':
                mfe = hp
                mae = lp
            if dir == 'short':
                mfe = -lp
                mae = -hp

            self.trades.append(
                {'订单': trade.ref,
                 '股票': trade.data._name,
                 # 'dir': dir,
                 '买入日期': datein,
                 '买价': round(pricein, 2),
                 '卖出日期': dateout,
                 '卖价': round(priceout, 2),
                 '收益率%': round(pcntchange, 2),
                 '利润': round(pnl, 2),
                 '利润总资产比%': round(pnlpcnt, 2),
                 '股数': size,
                 '股本': round(value, 2),
                 '仓位比%': round(value / total_value * 100, 2),
                 '累计收益': round(self.cum_profit, 2),
                 '持股天数': barlen,  # 以每根 bar 的时间为单位，这里按天计算
                 # 'pnl/bar': round(pbar, 2),
                 '最大利润%': round(mfe, 2),
                 '最大亏损%': round(mae, 2)})

class TradeRecord(bt.Analyzer):
    def __init__(self):
        self.history = []
        self.trades = []
        self.cumprofit = 0.0

    def notify_order(self, trade):

        self.current_trade = trade
        if not trade.isclosed:
            return
        record: Dict = self.get_trade_record(trade)
        self.trades.append(record)

    def stop(self):
        """统计最后一笔开仓未平仓的交易"""
        trade = self.current_trade

        if not trade.isopen:
            return

        record: Dict = self.get_trade_record(trade)
        self.trades.append(record)

    def get_trade_record(self, trade) -> Dict:

        brokervalue = self.strategy.broker.getvalue()
        dir = "long" if trade.history[0].event.size > 0 else "short"
        size = len(trade.history)
        barlen = trade.history[size - 1].status.barlen
        pricein = trade.history[size - 1].status.price
        datein = bt.num2date(trade.history[0].status.dt)

        is_close: int = size % 2  # 0表示偶数闭合 1表示奇数未闭合
        if is_close:

            # 交易闭合
            dateout = bt.num2date(trade.history[size - 1].status.dt)
            priceout = trade.history[size - 1].event.price
            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen + 1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen + 1))
            hp = 100 * (highest_in_trade - pricein) / pricein
            lp = 100 * (lowest_in_trade - pricein) / pricein

        else:
            # 交易没有闭合
            dateout = pd.to_datetime(trade.data.datetime.date(0))
            priceout = trade.data.close[0]
            hp = np.nan
            lp = np.nan
            barlen = np.nan

        if trade.data._timeframe >= bt.TimeFrame.Days:
            datein = datein.date()
            dateout = dateout.date()

        pcntchange = 100 * priceout / pricein - 100
        pnl = trade.history[size - 1].status.pnlcomm
        pnlpcnt = 100 * pnl / brokervalue

        pbar = pnl / barlen if barlen else np.nan
        self.cumprofit += pnl
        size = value = 0.0

        for record in trade.history:

            if abs(size) < abs(record.status.size):
                size = record.status.size
                value = record.status.value

        if dir == "long":
            mfe = hp
            mae = lp
        elif dir == "short":
            mfe = -lp
            mae = -hp

        return {
            "status": trade.status,  # 1-open,2-closed
            "ref": trade.ref,
            "ticker": trade.data._name,
            "dir": dir,
            "datein": datein,
            "pricein": pricein,
            "dateout": dateout,
            "priceout": priceout,
            "chng%": round(pcntchange, 2),
            "pnl": pnl,
            "pnl%": round(pnlpcnt, 2),
            "size": size,
            "value": value,
            "cumpnl": self.cumprofit,
            "nbars": barlen,
            "pnl/bar": round(pbar, 2),
            "mfe%": round(mfe, 2),
            "mae%": round(mae, 2),
        }

    def get_analysis(self):
        return self.trades


# 考虑佣金和印花税的股票百分比费用
class StockCommission(bt.CommInfoBase):
    params = (
        ("stamp_duty", 0.001),
        ("stocklike", True),  # 指定为股票模式
        ("commtype", bt.CommInfoBase.COMM_PERC),  # 使用百分比费用模式
        ("percabs", True),
    )  # commission 不以 % 为单位 # 印花税默认为 0.1%

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入时，只考虑佣金
            return abs(size) * price * self.p.commission
        elif size < 0:  # 卖出时，同时考虑佣金和印花税
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        else:
            return 0


class AddSignalData(bt.feeds.PandasData):
    """用于加载回测用数据

    添加信号数据
    """

    lines = ("rank",)

    params = (("rank", -1),)

def get_backtesting(
    data: pd.DataFrame,
    name: str = None,
    strategy: bt.Strategy = SignalStrategy,
    begin_dt: datetime.date = None,
    end_dt: datetime.date = None,
    **kw
) -> namedtuple:
    """回测

    添加了百分比滑点(0.0001)
    当日信号次日开盘买入
    Args:
        data (pd.DataFrame): OHLC数据包含信号
        name (str): 数据名称
        strategy (bt.Strategy): 策略

    Returns:
        namedtuple: result,cerebro
    """
    res = namedtuple("Res", "result,cerebro")

    # 如果是True则表示是多个标的 数据加载采用for加载多组数据
    mulit_add_data: bool = kw.get("mulit_add_data", False)
    # slippage_perc滑点设置
    slippage_perc: float = kw.get("slippage_perc", 0.0001)
    # 费用设置
    commission: float = kw.get("commission", 0.0002)
    stamp_duty: float = kw.get("stamp_duty", 0.001)
    # 是否显示log
    show_log: bool = kw.get("show_log", True)

    def LoadPandasFrame(data: pd.DataFrame) -> None:

        idx: np.ndarray = data.index.sort_values().unique()
        for code, df in data.groupby("code"):

            df = df.reindex(idx)
            df.sort_index(inplace=True)
            df = df[["open", "high", "low", "close", "volume","rank"]]
            df.loc[:, "volume"] = df.loc[:, "volume"].fillna(0)
            df.loc[:, ["open", "high", "low", "close"]] = df.loc[
                :, ["open", "high", "low", "close"]
            ].fillna(method="pad")

            datafeed = btfeeds.PandasData(dataname=df, fromdate=begin_dt, todate=end_dt)
            cerebro.adddata(datafeed, name=code)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1e9)
    if (begin_dt is None) or (end_dt is None):
        begin_dt = data.index.min()
        end_dt = data.index.max()
    else:
        begin_dt = pd.to_datetime(begin_dt)
        end_dt = pd.to_datetime(end_dt)
    #直接添加数据
    datafeed = AddSignalData(dataname=data, fromdate=begin_dt, todate=end_dt)
    cerebro.adddata(datafeed, name=name)
    #datafeed = datafeed = AddSignalData(dataname=data, fromdate=begin_dt, todate=end_dt,)
    #cerebro.adddata(datafeed, name='rank')
    #if mulit_add_data:
        #LoadPandasFrame(data)
    #else:
        #datafeed = AddSignalData(dataname=data, fromdate=begin_dt, todate=end_dt)
        #cerebro.adddata(datafeed, name=name)

    if slippage_perc is not None:
        # 设置百分比滑点
        cerebro.broker.set_slippage_perc(perc=slippage_perc)

    if (commission is not None) and (commission is not None):
        # 设置交易费用
        comminfo = StockCommission(commission=commission, stamp_duty=stamp_duty)

        cerebro.broker.addcommissioninfo(comminfo)

    # 添加策略
    cerebro.addstrategy(strategy, show_log=show_log)
    # 添加分析指标
    # 返回年初至年末的年度收益率
    cerebro.addanalyzer(bt.analyzers.Returns, _name="_Returns", tann=252)
    # 交易分析添加
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="_TradeAnalyzer")
    # 获取交易成本
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="_Transactions")
    # 计算交易统计
    cerebro.addanalyzer(bt.analyzers.PeriodStats, _name="_PeriodStats")
    # 返回收益率时序
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    # SQN
    cerebro.addanalyzer(bt.analyzers.SQN, _name="_SQN")
    # BuySell
    #cerebro.addanalyzer(bt.analyzers.BuySell, _name="_BuySell")
    # Share
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="_Sharpe",
        timeframe=bt.TimeFrame.Years,
        riskfreerate=0.04,
        annualize=True,
        factor=250,
    )

    # 这个需要在run开启tradehistory=True
    #cerebro.addanalyzer(TradeRecord, _name="_TradeRecord")
    cerebro.addanalyzer(TradeListAnalyzer, _name="_TradeListAnalyzer")
    # 添加自定义的 TradeLogger 分析器
    cerebro.addanalyzer(TradeLogger, _name='_trade_logger')
    cerebro.addanalyzer(TradeAnalyzer_1, _name='_TradeAnalyzer_1')
    cerebro.addanalyzer(OrderAnalyzer, _name='_OrderAnalyzer')


    result = cerebro.run(tradehistory=True)

    return res(result, cerebro)