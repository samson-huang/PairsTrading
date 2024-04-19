"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-01-11 10:03:20
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-01-11 11:29:09
Description: 策略
"""
import backtrader as bt
import pandas as pd

# 策略模板


class SignalStrategy(bt.Strategy):

    params = (
        ("open_threshold", 0.301),
        ("close_threshold", -0.301),
        ("show_log", True),
    )

    def log(self, txt, dt=None, show_log: bool = True):
        # log记录函数
        dt = dt or self.datas[0].datetime.date(0)
        if show_log:
            print(f"{dt.isoformat()}, {txt}")

    def __init__(self):

        self.dataclose = self.data.close
        self.signal = self.data.GSISI
        self.order = None

    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s"
                    % (
                        order.ref,  # 订单编号
                        order.executed.price,  # 成交价
                        order.executed.value,  # 成交额
                        order.executed.comm,  # 佣金
                        order.executed.size,  # 成交量
                        order.data._name,  # 股票名称
                    ),
                    show_log=self.p.show_log,
                )
            else:  # Sell
                self.log(
                    "SELL EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s"
                    % (
                        order.ref,
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm,
                        order.executed.size,
                        order.data._name,
                    ),
                    show_log=self.p.show_log,
                )

    def next(self):

        # 取消之前未执行的订单
        if self.order:
            self.cancel(self.order)

        if self.position:
            if (
                self.signal[0] <= self.params.close_threshold
                and self.signal[-1] <= self.params.close_threshold
            ):
                self.log("收盘价Close, %.2f" % self.dataclose[0], show_log=self.p.show_log)
                self.log(
                    "设置卖单SELL CREATE, %.2f信号为:%.2f,阈值为:%.2f"
                    % (self.dataclose[0], self.signal[0], self.params.close_threshold),
                    show_log=self.p.show_log,
                )
                self.order = self.order_target_value(target=0.0)

        elif (
            self.signal[0] >= self.params.open_threshold
            and self.signal[-1] >= self.params.open_threshold
        ):
            self.log("收盘价Close, %.2f" % self.dataclose[0], show_log=self.p.show_log)
            self.log(
                "设置买单 BUY CREATE, %.2f,信号为:%.2f,阈值为:%.2f"
                % (self.dataclose[0], self.signal[0], self.params.open_threshold),
                show_log=self.p.show_log,
            )
            self.order = self.order_target_percent(target=0.95)


class TopicStrategy(bt.Strategy):

    params = (
        ("show_log", True),
        ('percents', 0.2),
    )


    def __init__(self):
        self.lowest_stocks = []  # 用于存储收盘价格最低的5只股票


    def start(self):
        # 获取股票列表
        self.stocks = self.getdatanames()

    def next(self):
        # 每天开盘前执行
        if self.datas[0].open[0] > 0:  # 确保是开盘时间
            # 获取前一天的收盘价
            previous_closes = {stock:  self.getdatabyname(stock).close[-1] for stock in self.stocks}

            # 找出收盘价格最低的5只股票
            self.lowest_stocks = sorted(previous_closes, key=previous_closes.get)[:5]

            # 计算总投资金额
            total_percents = self.params.percents * self.broker.getcash()

            # 买入最低价格的5只股票
            for stock in self.lowest_stocks:
                if stock not in self.positions or self.positions[stock] == 0:
                    self.buy(stock, exectype=bt.Order.Market, size=total_percents / 5)

            # 卖出不在最低5只股票中的股票
            for stock in self.positions:
                if stock not in self.lowest_stocks:
                    self.sell(stock, exectype=bt.Order.Market)

    def stop(self):
        # 每天收盘后执行
        for stock in self.positions:
            self.close(stock)

class TopicStrategy345(bt.Strategy):
    params = (
        ("ranking_threshold", 5),
        ("show_log", True),
    )

    def __init__(self):
        # 初始化一个列表来存储当前持仓的数据源
        self.stocks = []
        # 初始化一个字典来存储每个数据源的目标持仓市值百分比
        self.target_positions = {}

    def next(self):
        # 获取当前日期的所有数据源的收盘价
        #ranks = [(data._name, data.rank[0]) for data in self.datas]
        #ranks =  [(self.data_names[data], data.rank[0]) for data in self.datas]
        ranks =  [self.data._name,self.data.rank]
        # 找出收盘价最高的5只股票
        top5 = sorted(ranks, key=lambda x: x[1], reverse=False)[:5]

        # 计算总资产
        total_assets = self.broker.getcash()

        # 设置目标持仓市值为总资产的20%
        for stock, rank in top5:
            # 计算目标持仓市值
            target_value = total_assets * 0.2

            # 获取当前持仓
            position = self.getposition(stock)

            # 如果当前持仓不足，买入股票
            if position is None or position.size < target_value / close:
                self.order_target_value(stock, target_value)
            # 如果当前持仓过多，卖出股票
            elif position.size > target_value / close:
                self.order_target_value(stock, target_value)
            # 记录目标持仓市值
            self.target_positions[stock] = target_value

    def log(self, txt, dt=None):
        ''' 用于记录日志的自定义方法 '''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        ''' 订单执行通知 '''
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            self.log(f'Order completed for {order.data._name}; executed at {order.executed.price}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order failed for {order.data._name}; status: {order.status}')


class LowRankStrategy(bt.Strategy):
    params = (
        ('buy_threshold', 5),  # 买入阈值
        ('stake', 0.2),  # 每只股票的仓位比例
        ("show_log", True),
    )

    def __init__(self):
        self.inds = {}  # 存储每只股票的指标数据
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]['prev_rank'] = d.rank(-1)  # 前一个交易日的收盘价

    def next(self):
        for d, ind in self.inds.items():
            pos = self.getposition(d).size
            if pos == 0:
                # 当前无头寸
                if ind['prev_rank'][0] <= self.params.buy_threshold:
                    # 买入信号
                    #size = int(self.broker.get_cash() * self.params.stake / d.close[0])
                    self.order = self.order_target_percent(data=d, target=0.2)
                    print(f'Buy {d._name}, Size: 0.2, Prev Close: {ind["prev_rank"][0]:.2f}')
            else:
                # 当前有头寸
                if ind['prev_rank'][0] > self.params.buy_threshold:
                    # 卖出信号
                    self.order = self.order_target_percent(data=d, target=0.0)
                    print(f'Sell {d._name}, Size: {pos}, Prev Close: {ind["prev_rank"][0]:.2f}')

    def stop(self):
        print('Strategy completed')


