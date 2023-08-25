from qlib.data import D
df = D.features(
    ["SH512690"],
    ["$open", "$high", "$low", "$close", "$Volume"],
    start_time="2023-01-01",
    end_time="2023-08-24",
)

# 转换列名为matplotlib要求的命名
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# 计算涨跌幅
df['PCT_Change'] = (df['Close'] - df['Open'])/df['Open']


# 创建图形
fig = plt.figure(figsize=(16,8))

# 在图形上创建2个子图
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# 绘制蜡烛图
df = df.reset_index(level=0, drop=True)
ax1.plot(df.index, df['Close'], color='g')
ax1.gca().get_yaxis().get_major_formatter().set_scientific(False)
ax1.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
ax1.title('SH512690 蜡烛图')
ax1.xticks(rotation=90)
ax1.grid(linestyle="--")

for index, row in df.iterrows():
    if row['Open'] > row['Close']:
        plt.bar(index, row['High']-row['Low'], bottom=row['Close'], color='r')
    else:
        plt.bar(index, row['High']-row['Low'], bottom=row['Open'], color='g')




df_signal.loc[['2023-01-04'], :].sort_values(by='score', ascending=False).head()
df_signal.to_csv("c:\\temp\\df_signal_20230825.csv")
df_instrument.to_csv("c:\\temp\\score_20230825.csv")
# 读取score数据
scores = df_instrument[df_instrument['instrument'] == 'SH512690']['score'].tolist()

# 在图中绘制score数据
ax2.plot(df_instrument['datetime'], scores, color='r', linestyle='--', marker='o')
ax2.xticks(rotation=90)
# 设置图例
ax2.legend(['Close', 'Score'])

plt.tight_layout()
plt.show()

















# 创建图形
fig = plt.figure(figsize=(16,8))

# 在图形上创建2个子图
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# 绘制蜡烛图
df = df.reset_index(level=0, drop=True)
ax1.plot(df.index, df['Close'], color='g')
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('SH512690 蜡烛图')
plt.xticks(rotation=90)
plt.grid(linestyle="--")

for index, row in df.iterrows():
    if row['Open'] > row['Close']:
        plt.bar(index, row['High']-row['Low'], bottom=row['Close'], color='r')
    else:
        plt.bar(index, row['High']-row['Low'], bottom=row['Open'], color='g')


scores = df_instrument[df_instrument['instrument'] == 'SH512690']['score'].tolist()

# 在图中绘制score数据
ax2.plot(df_instrument['datetime'], scores, color='r', linestyle='--', marker='o')
# 设置图例
ax2.legend(['Close', 'Score'])

plt.tight_layout()
plt.show()

