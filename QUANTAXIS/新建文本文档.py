import QUANTAXIS as QA

user = QA.QA_User(username ='quantaxis', password = 'quantaxis')
portfolio=user.new_portfolio('x1')
account = portfolio.new_account(account_cookie='test')