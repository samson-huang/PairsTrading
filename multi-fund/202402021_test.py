import sys
import pandas as pd
sys.path.append('C://Local_library')
from hugos_toolkit.BackTestTemplate import get_backtesting,AddSignalData
from hugos_toolkit.BackTestTemplate import TopicStrategy


df = pd.read_csv('c:\\temp\\ranked_data_20240221_1.csv',parse_dates=['datetime'],index_col=0)


bt_result = get_backtesting(
    df,
    strategy=TopicStrategy,
    mulit_add_data=True,
    feedsfunc=AddSignalData,
    strategy_params={"selnum": 5, "pre": 0.05,'ascending':False,'show_log':False},
)