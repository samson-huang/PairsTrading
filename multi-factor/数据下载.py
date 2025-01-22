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

