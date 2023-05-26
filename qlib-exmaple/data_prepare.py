def GetDataFromTushare(self, today, fpath):
    pro = ts.pro_api()
    df = pro.daily(trade_date=today)
    for index, row in df.iterrows():
        code = row['ts_code']
        sp = code.index(".")
        ncode = code[sp + 1:].lower() + code[:sp]
        trade_date = str(row['trade_date'])
        date = trade_date[0:4] + '-' + trade_date[4:6] + '-' + trade_date[6:8]
        topen = row['open']
        thigh = row['high']
        tlow = row['low']
        tclose = row['close']
        tvolume = row['vol']

        value = [ncode, date, topen, tlow, thigh, tclose, tvolume, tclose]
        names = ['symbol', 'date', 'open', 'low', 'high', 'close', 'volume', 'adjclose']

        fname = fpath + '/day_csv/' + ncode + ".csv"
        if not os.path.exists(fname):
            fo = open(fname, "w+")
            fo.write(",".join(names) + "\n")
            valueline = ",".join([str(i) for i in value])
            fo.write(valueline + "\n")
            fo.close()
        else:
            fo = open(fname, "a")
            valueline = ",".join([str(i) for i in value])
            fo.write(valueline + "\n")
            fo.close()

#########################################################
#################################################################
###################################################################
INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"
   def GetDataFromEastMoney(self，days, fpath):
        print ("start to download index data")
        _format = "%Y%m%d"
        _begin = (datetime.now() - timedelta(days=days)).strftime(_format)
        _end   = (datetime.now() - timedelta(days=1)).strftime(_format)
        for _index_name, _index_code in {"csi300": "000300", "csi100": "000903"}.items():
            print (f"get bench data: {_index_name}({_index_code})......")
            print (INDEX_BENCH_URL.format(index_code=_index_code, begin=_begin, end=_end))
            try:
                df = pd.DataFrame(
                    map(
                        lambda x: x.split(","),
                        requests.get(INDEX_BENCH_URL.format(index_code=_index_code, begin=_begin, end=_end)).json()[
                            "data"
                        ]["klines"],
                    )
                )
            except Exception as e:
                print (f"get {_index_name} error: {e}")
                continue
            df.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.astype(float, errors="ignore")
            df["adjclose"] = df["close"]
            df["symbol"] = f"sh{_index_code}"
            ncode = f"sh{_index_code}"
            fname = fpath + '/day_csv/' + ncode + ".csv"
            df.to_csv(fname, index=False)


#dump_update ==增量==更新数据
#dump_all ==全量==覆盖数据
python qlib-main/scripts/dump_bin.py dump_update --csv_path  data/day_csv --qlib_dir ~/.qlib/qlib_data/cn_data --include_fields  open,close,high,low,volume,factor




