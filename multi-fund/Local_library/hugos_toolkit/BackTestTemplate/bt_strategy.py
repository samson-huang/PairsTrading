"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-01-11 10:03:20
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-01-11 11:29:09
Description: 策略
"""
import backtrader as bt

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
        ("ranking_threshold", 5),
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
        self.signal = self.data.rank
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

        # 获取当前策略中的资金
        cash = self.broker.getcash()

        # 根据排名选择前五名股票
        #top_stocks = sorted(self.signal[0], key=lambda x: x[1], reverse=True)[:5]

        # 遍历当前持有的股票
        for stock in list(self.positions.keys()):
            # 如果股票不在前五名，则卖出
            if self.signal[-1]>ranking_threshold:
                self.log("收盘价Close, %.2f" % self.dataclose[0], show_log=self.p.show_log)
                self.log(
                    "设置卖单SELL CREATE, %.2f信号为:%.2f,阈值为:%.2f"
                    % (self.dataclose[0], self.signal[0], self.params.ranking_threshold),
                    show_log=self.p.show_log,
                )
                self.order = self.order_target_value(target=0.0)

            if stock not in self.positions and self.signal[-1]<=ranking_threshold:
                # 计算可以购买的股票数量
                #buy_amount = cash // (stock.price * 0.2)
                # 买入股票，使其占用的资金不超过总资产的20%
                self.log("收盘价Close, %.2f" % self.dataclose[0], show_log=self.p.show_log)
                self.log(
                    "设置买单 BUY CREATE, %.2f,信号为:%.2f,阈值为:%.2f"
                    % (self.dataclose[0], self.signal[0], self.params.ranking_threshold),
                    show_log=self.p.show_log,
                )
                self.order = self.order_target_percent(target=0.2)



class TopPerformersStrategy(bt.Strategy):
    def __init__(self):
        # 初始化一个字典来存储每只股票的排名
        self.signal = {}
        self.order = None

    def log(self, txt, dt=None, show_log: bool = True):
        # log记录函数
        dt = dt or self.datas[0].datetime.date(0)
        if show_log:
            print(f"{dt.isoformat()}, {txt}")

    def set_signal(self, stock, rank):
        # 设置或更新股票的排名
        self.signal[stock] = rank

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
        # 获取当前策略中的资金
        cash = self.broker.getcash()

        # 根据排名选择前五名股票
        top_stocks = sorted(self.signal.items(), key=lambda x: x[1], reverse=True)[:5]

        # 遍历当前持有的股票
        for stock in list(self.positions.keys()):
            # 如果股票不在前五名，则卖出
            if stock not in top_stocks:
                self.close(stock)

        # 遍历前五名股票
        for stock, _ in top_stocks:
            # 检查是否已经持有该股票
            if stock not in self.positions:
                # 计算可以购买的股票数量
                buy_amount = cash // (stock.price * 0.2)
                # 买入股票，使其占用的资金不超过总资产的20%
                self.buy(stock, size=buy_amount, exectype=bt.Order.Market)

