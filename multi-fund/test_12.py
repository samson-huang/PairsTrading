import qlib
from qlib.constant import REG_CN
import pandas as pd
provider_uri = "C:/Users/huangtuo/.qlib/qlib_data/fund_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data import D
from typing import List, Tuple, Dict
from qlib.contrib.data.handler import Alpha158


def _get_score_ic_test(pred_label: pd.DataFrame):
    """

    :param pred_label:
    :return:
    """
    concat_data = pred_label.copy()
    concat_data.dropna(axis=0, how="any", inplace=True)
    _ic = concat_data.groupby(level=['datetime']).apply(
        lambda x: x["label"].corr(x["score"])
    )
    #_rank_ic = concat_data.groupby(level=['datetime']).apply(
    #    lambda x: x["label"].corr(x["score"], method="spearman")
    #)
    #return pd.DataFrame({"ic": _ic, "rank_ic": _rank_ic})
    return pd.DataFrame({"ic": _ic})

def _get_score_ic_frame(fetch_factor: pd.DataFrame):
    columns_counts = len(fetch_factor.columns)
    # 取df第一个索引层级的'datetime'值
    dates = bench.index.get_level_values(0)
    # 新建df
    score_ic_frame = pd.DataFrame(index=[])
    # 赋值给df的索引
    score_ic_frame.index = dates
    for i in range(columns_counts - 1):
        # 以KLEN参数为例子进行测算。
        fetch_factor_one = fetch_factor.iloc[:, i]
        # 取columns的名称
        columns_names = fetch_factor.columns[i]
        : pd.DataFrame = pd.concat((next_ret, fetch_factor_one), axis=1)
        feature_df.columns = pd.MultiIndex.from_tuples(
            [("label", "next_ret"), ("feature", columns_names)]
        )

        score_df: pd.DataFrame = feature_df.dropna().copy()
        score_df.columns = ['label', 'score']
        get_score_ic = _get_score_ic_test(score_df)
        # 替换单个列名
        get_score_ic = get_score_ic.rename(columns={'ic': columns_names})
        score_ic_frame = pd.concat([score_ic_frame, get_score_ic], axis=1)
    return score_ic_frame

#主函数
if __name__ == '__main__':
    test_period = ("2019-01-01", "2024-05-15")
    market = "filter_fund"
    benchmark = "SZ160706"

    dh = Alpha158(instruments='filter_fund',
                  start_time=test_period[0],
                  end_time=test_period[1],
                  infer_processors={}
                  )



    POOLS: List = D.list_instruments(D.instruments(market), as_list=True)

    # 未来期收益
    next_ret: pd.DataFrame = D.features(POOLS, fields=["Ref($close,-1)/$close-1"], start_time=test_period[0],
                                        end_time=test_period[1], freq='day')
    next_ret.columns = ["next_ret"]
    next_ret: pd.DataFrame = next_ret.swaplevel()
    next_ret.sort_index(inplace=True)

    # 基准
    bench: pd.DataFrame = D.features([benchmark], fields=["$close/Ref($close,1)-1"], start_time=test_period[0],
                                     end_time=test_period[1])
    bench: pd.Series = bench.droplevel(level=0).iloc[:, 0]
    #######################################################################
    # 生成单日每个因子的IC数值
    fetch_factor = dh.fetch()

    test2: pd.DataFrame = _get_score_ic_frame(fetch_factor)