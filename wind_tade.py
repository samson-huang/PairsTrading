from WindPy import *
w.start()

class MyOrder(): 
      #def __init__(self):
            #"退出下单"按钮事件
    def login(self):
                
        LoginID=w.tlogon("0000","0","w1461511901","login123","SHSZ")
        print(LoginID)
      #登陆账户, 返回登陆ID
    def windorder(self,stock_code,trade_side,amount):       
        price=w.wsq(stock_code,'rt_last').Data[0]
        windorder_return=w.torder(stock_code,trade_side,price,amount,logonid=1)
        print(windorder_return)	
    #登出多账户
    def logout(self):
        w.tlogout()
        self.close()
	
if __name__ == "__main__":
    
    order_excel = MyOrder()
    order_excel.login()
    order_excel.windorder()
    #order_excel.logout()



#查询登录账号
#w.tquery("LoginID")

#查询资金情况
#w.tquery("Order",logonid=1)

#查委托情况
#w.tquery("Order",logonid=1)

#查当然成交情况
#w.tquery("Trade",logonid=1)










w.tlogout(0)


