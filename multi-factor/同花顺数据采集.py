import iFinDPy

# 登录函数
def thslogindemo():
    # 输入用户的帐号和密码
    thsLogin = iFinDPy.THS_iFinDLogin("zdzq320","*****************")
    print(thsLogin)
    if thsLogin != 0:
        print('登录失败')
    else:
        print('登录成功')


thslogindemo()

def get_hs300_weights_old(start_year, end_year):
    """
    获取指定年份范围内每年 6 月 30 日和 12 月 31 日沪深 300 成分股权重数据

    :param start_year: 起始年份
    :param end_year: 结束年份
    :param account: 同花顺 iFinD 账号
    :param password: 同花顺 iFinD 密码
    :return: 包含权重数据的 DataFrame，若未获取到有效数据则返回 None
    """
    weights_data = []

    for year in range(start_year, end_year + 1):
        # 获取 1 月 01 日的数据
        date_1 = f"{year}0101"
        # 拆分查询参数
        paramname_1 = f"date={date_1};index_name=000300.SH"
        fields_1 = "p03563_f001:Y,p03563_f002:Y,p03563_f003:Y,p03563_f004:Y"
        funoption_1 = "format:dataframe"
        result_1 = iFinDPy.THS_DR('p03563', paramname_1, fields_1, funoption_1)
        if result_1.errorcode == 0:
            df_1 = result_1.data
            df_1['date'] = date_1
            weights_data.append(df_1)
        else:
            print(f"获取 {date_1} 数据失败，错误代码: {result_1.errorcode}")

        # 获取 12 月 31 日的数据
        date_2 = f"{year}1231"
        # 拆分查询参数
        paramname_2 = f"date={date_2};index_name=000300.SH"
        fields_2 = "p03563_f001:Y,p03563_f002:Y,p03563_f003:Y,p03563_f004:Y"
        funoption_2 = "format:dataframe"
        result_2 = iFinDPy.THS_DR('p03563', paramname_2, fields_2, funoption_2)
        if result_2.errorcode == 0:
            df_2 = result_2.data
            df_2['date'] = date_2
            weights_data.append(df_2)
        else:
            print(f"获取 {date_2} 数据失败，错误代码: {result_2.errorcode}")


    # 合并所有数据
    if weights_data:
        all_weights = pd.concat(weights_data, ignore_index=True)
        return all_weights
    else:
        print("未获取到有效数据")
        return None

def get_hs300_weights(start_date, end_date):
    """
    获取指定日期范围内每天沪深 300 成分股权重数据

    :param start_date: 起始日期，格式为 'YYYYMMDD'
    :param end_date: 结束日期，格式为 'YYYYMMDD'
    :return: 包含权重数据的 DataFrame，若未获取到有效数据则返回 None
    """
    weights_data = []
    from datetime import datetime, timedelta

    # 将输入的日期字符串转换为 datetime 对象
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    # 循环遍历从起始日期到终止日期的每一天
    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y%m%d')
        # 拆分查询参数
        paramname = f"date={date_str};index_name=000300.SH"
        fields = "p03563_f001:Y,p03563_f002:Y,p03563_f003:Y,p03563_f004:Y"
        funoption = "format:dataframe"
        result = iFinDPy.THS_DR('p03563', paramname, fields, funoption)
        if result.errorcode == 0:
            df = result.data
            df['date'] = date_str
            weights_data.append(df)
        else:
            print(f"获取 {date_str} 数据失败，错误代码: {result.errorcode}")

        # 日期加一天
        current_date += timedelta(days=1)

    # 合并所有数据
    if weights_data:
        all_weights = pd.concat(weights_data, ignore_index=True)
        return all_weights
    else:
        print("未获取到有效数据")
        return None

import pandas as pd
# 示例调用
start_date = '20050101'
end_date = '20050206'
#account = 'your_account'
#password = 'your_password'
weights_df = get_hs300_weights(start_date, end_date)

iFinDPy.THS_iFinDLogout()


weights_df=pd.read_csv("c:\\temp\\weights_df_20160616.csv")
weights_df = weights_df.drop(weights_df.columns[[0]], axis=1)
weights_df_1=pd.read_csv("c:\\temp\\weights_df_1_20160617.csv")
weights_df_1 = weights_df_1.drop(weights_df_1.columns[[0]], axis=1)


new_weights_df = pd.concat([weights_df, weights_df_1],axis=0)