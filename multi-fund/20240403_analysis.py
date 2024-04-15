import qlib
from qlib.constant import REG_CN
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from datetime import datetime
#ranked_data.to_csv("c:\\temp\\ranked_data_20240221.csv")
ranked_data = pd.read_csv('c:\\temp\\ranked_data_all.csv', parse_dates=['datetime'], index_col='datetime')
#导入hugos_toolkit库需要指定目录
import sys
sys.path.append('C://Local_library')
from hugos_toolkit.BackTestTemplate import TopicStrategy,get_backtesting,AddSignalData
from hugos_toolkit.BackTestReport.tear import analysis_rets



#主函数
if __name__ == '__main__':
    bt_result = get_backtesting(
        ranked_data,
        strategy=TopicStrategy,
        mulit_add_data=True,
        feedsfunc=AddSignalData,
        strategy_params={"selnum": 5, "pre": 0.05, 'ascending': False, 'show_log': False},
    )
    trade_logger = bt_result.result[0].analyzers._trade_logger.get_analysis()
    TradeListAnalyzer = bt_result.result[0].analyzers._TradeListAnalyzer.get_analysis()

    OrderAnalyzer = bt_result.result[0].analyzers._OrderAnalyzer.get_analysis()
    # 将 trades 列表转换为 DataFrame
    #trade_logger_df = pd.DataFrame(trade_logger['trades'])

    # 可选: 设置列名
    #rade_logger_df.columns = ['ref', 'buy_date', 'buy_price', 'buy_size', 'sell_date', 'sell_price', 'sell_size', 'pnl']

    # 打印 DataFrame
    #print(OrderAnalyzer)
    # 将订单信息列表转换为 DataFrame
    trader_df = pd.DataFrame(OrderAnalyzer)
    orders_df = pd.DataFrame(trade_logger)
    # 设置列名
    #orders_df.columns = ['ref', 'status', 'size', 'price', 'value', 'reason', 'date', 'data', 'type']

    # 打印 DataFrame
    print(trader_df)
    print(orders_df)