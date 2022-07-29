#!/usr/bin/env python
#-*-coding:utf-8-*-

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

_user = "375317196@qq.com"
_pwd = "kkoctnqsxjxabhdb"
# _to = "ft_clover@163.com"
_recer=["tuo.huang@zdzq.com","huangtuo02@163.com",]


#创建一个带附件的实例
msg=MIMEMultipart()
msg["Subject"] = " don't panic"
msg["From"] = _user
msg["To"] = ",".join(_recer)#区别与给一个人发，指定某个人用 msg["To"] = _to 多个人用.join


#邮件正文内容
msg.attach(MIMEText('假如你立志要能言善辩，请先学会专注聆听。做一个有趣的人，并对他人感兴趣。问对方乐于回答的问题，鼓励他们谈论自己的经历。'
                    '请记住，你的谈话对象并不关心你和你的问题，而对他们自己、他们的欲望和烦恼要感兴趣得多。他的牙疼远比异国饿殍遍地的饥荒更重要，他脖子上的疖子也远比非洲的四十次地震更让人心烦。所以下次开口之前，请先想想这一点。','plain', 'utf-8'))
#构造附件1，传输当前目录下的图片.txt文件
att1=MIMEText(open('jmeter.txt','rb').read(),'base64','utf-8')
att1['Content-Type']='application/octet-stream'
att1['Content-Disposition']='attachment;filename="demo.txt"' #filename 填什么，邮件里边展示什么

#构造附件2，传输当前目录下的图片.jpg文件
att2=MIMEText(open('图片.jpg','rb').read(),'base64','utf-8')
att2['Content-Type']='application/octet-stream'
att2['Content-Disposition']='attachment;filename="demo.jpg"' #filename填什么，邮件里边展示什么

msg.attach(att1)
msg.attach(att2)


try:
    s=smtplib.SMTP_SSL("smtp.qq.com",465)
    s.login(_user,_pwd)
    s.sendmail(_user,_recer,msg.as_string())
    s.quit()
    print("Success!")
except smtplib.SMTPException as e:
    print("Failed,%s"%e)