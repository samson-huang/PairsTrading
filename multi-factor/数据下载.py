import akshare as ak
import pandas as pd
import time
import random
# 获取行业信息数据框东财85个行业
stock_board_industry_name_em_df = ak.stock_board_industry_name_em()
# 提取所有行业代码
industry_codes = stock_board_industry_name_em_df['板块名称'].tolist()

# 初始化一个空列表来存储每个行业的结果
results = []

# 遍历所有行业代码，调用函数并存储结果
for code in industry_codes:
    try:
        cons_df = ak.stock_board_industry_cons_em(symbol=code)
        cons_df['板块名称'] = code  # 将code值作为一列添加到cons_df中
        results.append(cons_df)
                # 随机暂停1到10秒
        time.sleep(random.uniform(1, 5))
    except Exception as e:
        print(f"Error processing {code}: {e}")

# 合并所有结果到一个大的DataFrame中
combined_df = pd.concat(results, ignore_index=True)
combined_df.to_csv("c:\\temp\\20250122.csv")

#导入数据
import pandas as pd
combined_df=pd.read_csv('c:\\temp\\20250122.csv')
combined_df = combined_df.drop(combined_df.columns[[0, 1]], axis=1)
combined_df.head()

#导入总市值，及其他相关
#上海市场
stock_sh_a_spot_em_df = ak.stock_sh_a_spot_em()
#深圳市场
stock_sz_a_spot_em_df = ak.stock_sz_a_spot_em()
#all所有股票
stock_all = pd.concat([stock_sh_a_spot_em_df, stock_sz_a_spot_em_df], ignore_index=True)
###
# 计算总市值单价并取整数
stock_all['总股本'] = (stock_all['总市值'] / stock_all['最新价']).astype(int)
stock_all['流通股本'] = (stock_all['总市值'] / stock_all['最新价']).astype(int)

stock_all.to_csv("c:\\temp\\stock_all_20250124.csv")
#stock_all=pd.read_csv('c:\\temp\\20250122.csv')
#stock_all = stock_all.drop(stock_all.columns[[0, 1]], axis=1)

# 按代码列合并两个DataFrame，并添加一个指示列
stock_all['代码'] = stock_all['代码'].astype(str)
combined_df['代码'] = combined_df['代码'].astype(str).str.zfill(6)

# 找出只存在于df1中的数据,结果都是退市股票或者B股
only_in_df1 = stock_all[~stock_all['代码'].isin(combined_df['代码'])]


#加上总市值，流通市值
# 选择result_all中的总市值和流通市值列，并按代码列匹配
result_to_add = stock_all[['代码', '总股本', '流通股本']]

# 使用merge函数将总市值和流通市值列添加到combined_df中
combined_df_all = pd.merge(combined_df, result_to_add, on='代码', how='left')

#取字段
combined_df_new=combined_df_all[['代码','名称','板块名称', '市盈率-动态', '市净率','总股本', '流通股本']]

# 定义一个函数，根据代码的开头添加市场标识
def add_market_code(code):
    if str(code).startswith('6'):
        return f"SH{code}"
    elif str(code).startswith('0'):
        return f"SZ{code}"
    elif str(code).startswith('3'):
        return f"SZ{code}"
    else:
        return code  # 如果代码不符合6或0开头，直接返回原代码

# 应用函数到代码列
combined_df_new['代码'] = combined_df_new['代码'].apply(add_market_code)
combined_df_new.to_csv('c:\\temp\\combined_df_new_20250124.csv')
#############################
#combined_df_new=pd.read_csv('c:\\temp\\combined_df_new_20250124.csv')
#combined_df_new = combined_df_new.drop(combined_df_new.columns[[0]], axis=1)
#invalid_codes = combined_df_new[~combined_df_new['代码'].str.contains('SH|SZ')]
#combined_df_new[combined_df_new['代码'] == 'SH600823']

#############################





########################################################
#合并后发行有些退市股票没有行业标识，需要重新处理股票行业分类数据
empty_sector_rows = weights_df_1[pd.isna(weights_df_1['板块名称'])]
grouped_counts = empty_sector_rows.groupby(['名称', '代码']).size().reset_index(name='条数')

#我手动处理一下退市股票行业
########################################################
grouped_counts=pd.read_csv("c:\\temp\\grouped_counts_20250206.csv")
grouped_counts = grouped_counts.rename(columns={
    "行业名称": "板块名称"})

# 确保 grouped_counts 包含所有需要的列
grouped_counts = grouped_counts[['名称', '代码', '板块名称']]
grouped_counts['市盈率-动态'] = None  # 添加新列并填充空值
grouped_counts['市净率'] = None
grouped_counts['总股本'] = None
grouped_counts['流通股本'] = None

# 合并两个 DataFrame
combined_df_new = pd.concat([combined_df_new, grouped_counts], ignore_index=True)

#获取指数6月底跟12月底的权重数据，取数方式见“同花顺数据采集.py”
#########################################################################
weights_df=pd.read_csv("c:\\temp\\weights_df_20160616.csv")
weights_df = weights_df.drop(weights_df.columns[[0]], axis=1)
weights_df_1=pd.read_csv("c:\\temp\\weights_df_1_20160617.csv")
weights_df_1 = weights_df_1.drop(weights_df_1.columns[[0]], axis=1)


new_weights_df = pd.concat([weights_df, weights_df_1],axis=0)

new_weights_df.rename(columns={'p03563_f001':'取值日期',
                                  'p03563_f002': '代码',
                                  'p03563_f003': '名称',
                                  'p03563_f004': '权重'}, inplace=True)

weights_df=new_weights_df
weights_df['代码'] = weights_df['代码'].apply(lambda x: x[-2:] + x[:-3])
#假设 combined_df_new 和 weights_df 是你的数据框
# 首先确保两个数据框的“代码”列都是字符串格式，以确保正确匹配
combined_df_new['代码'] = combined_df_new['代码'].astype(str)
weights_df['代码'] = weights_df['代码'].astype(str)

# 根据“代码”列进行合并，使用左连接（left join）以保留 weights_df 中的所有行
weights_df_1 = pd.merge(weights_df, combined_df_new[['代码', '板块名称']], on='代码', how='left')



########################################
#取行情数据，直接取，其实可以重qlib里取
########################################
ranked_data = pd.read_csv('c:\\temp\\ranked_data_20240704.csv',
                          parse_dates=['datetime'],
                          index_col=['datetime', 'code'])

# 使用 merge 函数将 combined_df_new 中的相关列合并到 ranked_data 中
ranked_data_1 = ranked_data.join(combined_df_new.set_index('代码'), on='code', how='left')

#生成总市值
ranked_data_1['market_cap']=ranked_data_1['close']*ranked_data_1['总股本']
ranked_data_1 = ranked_data_1.rename(columns={
    "板块名称": "INDUSTRY_CODE"})

#取字段
ranked_data_2=ranked_data_1[['rank','INDUSTRY_CODE','market_cap']]

ranked_data_2 = ranked_data_2.rename_axis(index={'datetime': 'date'})

#取字段
ranked_data_2=ranked_data_1[['close','rank','INDUSTRY_CODE','market_cap']]
#
ranked_data_2['NEXT_RET'] = ranked_data_2['close'].pct_change().shift(-1)

ranked_data_2 = ranked_data_2.rename(columns={
    "rank": "SCORE"})

# 按 'datetime' 分组，并使用 'market_cap' 的中位数进行回填
ranked_data_2['market_cap'] = ranked_data_2.groupby('datetime')['market_cap'].transform(lambda x: x.fillna(x.median()))

#查看一下数据
#ranked_data_2.xs('SH600001', level='code')

 查看多级索引的描述信息
if isinstance(ranked_data_2.index, pd.MultiIndex):
    for level_index, level in enumerate(ranked_data_2.index.levels):
        print(f"Level {level_index}:")
        print(f"- Unique values: {level.nunique()}")
        print(f"- First value: {level[0]}")
        print(f"- Last value: {level[-1]}")
        print(f"- Size: {len(level)}\n")
else:
    print("Index is not a MultiIndex.")


######合并权重数据#############


weights_df_1 = weights_df_1.rename(columns={
    "date": "datetime",
    "代码": "code",
    "权重": "weight",})

#取字段
weights_df_1=weights_df_1[['datetime','code','名称','weight']]


# 将 ranked_data_2 的索引重置为普通列
ranked_data_2 = ranked_data_2.reset_index()
# 将 weights_df_1 的 datetime 列转换为 datetime64[ns] 类型
weights_df_1['datetime'] = pd.to_datetime(weights_df_1['datetime'].astype(str), format='%Y%m%d')
# 合并两个 DataFrame
merged_data = pd.merge(ranked_data_2, weights_df_1[['datetime', 'code', 'weight']], on=['datetime', 'code'], how='left')
# 将 datetime 和 code 设置回索引
merged_data.set_index(['datetime', 'code'], inplace=True)



