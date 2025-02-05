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

def get_hs300_weights(start_year, end_year):
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
        # 获取 6 月 30 日的数据
        date_1 = f"{year}0630"
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

import pandas as pd
# 示例调用
start_year = 2005
end_year = 2024
#account = 'your_account'
#password = 'your_password'

weights_df = get_hs300_weights(start_year, end_year)


iFinDPy.THS_iFinDLogout()