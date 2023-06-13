from pprint import pprint
from pathlib import Path
import pandas as pd
import qlib


import tushare as ts
import warnings
import json
warnings.filterwarnings('ignore')
# 请根据自己的情况填写ts的token
setting = json.load(open('C://config//config.json'))
# pro  = foundation_tushare.TuShare(setting['token'], max_retry=60)
ts.set_token(setting['token'])
pro = ts.pro_api()

qlib.init()

MARKET = "csi300"
BENCHMARK = "SH000300"
EXP_NAME = "tutorial_exp"

hd=pd.read_csv('c://temp/Alpha158_test.csv')
hd.head()
hd.set_index(['datetime', 'instrument'], inplace=True)
#修改索引名字
new_names = ['date', 'code']
hd.index.set_names(new_names, inplace=True)


hd.loc[:, ['KMID','KLEN']].head()

# 获取申万二级行业列表
SW2021 = pro.index_classify(level='L2', src='SW2021')

# 初始化一个字典来存储每个行业的股票代码
industry_stocks = {}

# 遍历二级行业列表
for _, row in SW2021.iterrows():
    industry_code = row['index_code']
    industry_name = row['industry_name']

    # 获取指定行业的股票列表
    stocks = pro.index_member(index_code=industry_code)

    # 提取股票代码，并添加到字典中
    stock_codes = list(stocks['con_code'])
    industry_stocks[industry_name] = stock_codes

# 打印各行业的股票代码
# 将字典转换为 DataFrame，并添加行业名称列
df = pd.DataFrame([(k, v) for k, val in industry_stocks.items() for v in val], columns=['industry_name', 'stock_code'])

df = df.rename(columns={'stock_code': 'code', 'industry_name': 'industry_code'})

df = df.dropna(axis=0, how='any')
df = df.drop_duplicates(subset=['code'])
df.describe()
# 将 A 列设置为索引
industry_code=df
industry_code['code'] = industry_code['code'].str[7:9]+industry_code['code'].str[0:6]
industry_code = industry_code.set_index('code')


# 合并两个 DataFrame，并根据 date 和 code 进行排序
merged_df = pd.merge(hd, industry_code, left_index=True, right_index=True).sort_index()




