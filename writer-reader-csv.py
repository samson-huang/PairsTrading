#输出数据写入CSV文件
import csv
data = [
    ("SH", "600000.SH", 1000,'buy'),
    ("SZ", "000001.SZ", 1000,'buy'),
    ("SH", "600004.SH", 1000,'buy')
]

#Python3.4以后的新方式，解决空行问题
with open('c://excel-export//write.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for list in data:
        print(list)
        csv_writer.writerow(list)

       
#读取csv文件内容
import csv
stock_code  = []
trade_side  = []
reader = csv.reader(open("c://excel-export//write.csv"))
#csv中有四列数据，遍历读取时使用四个变量分别对应
for market, code, amount,tradeside in reader:
  stock_code.append(code)
  trade_side.append(tradeside)
  print(market, "; ",  code , "; ",  amount,"; ",  tradeside)

print(stock_code)
print(trade_side)




##########直接下单###############


import csv
stock_code  = []
trade_side  = []
reader = csv.reader(open("c://excel-export//write.csv"))
#csv中有三列数据，遍历读取时使用三个变量分别对应
for market, code, amount,tradeside in reader:
  stock_code.append(code)
  trade_side.append(tradeside)
  print(market,"; ",code,"; ",amount,"; ",tradeside)

print(stock_code)
print(trade_side)


order_excel = MyOrder()
order_excel.login()
order_excel.windorder(stock_code,trade_side)