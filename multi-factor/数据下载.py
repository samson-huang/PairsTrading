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
#invalid_codes = combined_df_new[~combined_df_new['代码'].str.contains('SH|SZ')]
#############################
ranked_data = pd.read_csv('c:\\temp\\ranked_data_20240704.csv',
                          parse_dates=['datetime'],
                          index_col=['datetime', 'code'])

# 使用 merge 函数将 combined_df_new 中的相关列合并到 ranked_data 中
ranked_data_1 = ranked_data.join(combined_df_new.set_index('代码'), on='code', how='left')

#生成总市值
ranked_data_1['market_cap']=ranked_data_1['close']*ranked_data_1['总股本']





