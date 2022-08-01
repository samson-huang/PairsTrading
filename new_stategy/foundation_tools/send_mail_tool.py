#!/usr/bin/env python
#-*-coding:utf-8-*-

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import json
import warnings
warnings.filterwarnings('ignore')
# 请根据自己的情况填写ts的token
setting = json.load(open('C://config//config.json'))


_user = "375317196@qq.com"
_pwd = setting['qq']
# _to = "ft_clover@163.com"
_recer=["tuo.huang@zdzq.com","huangtuo02@163.com",]

class send_mail_tool:
    def __init__(self, _user=_user, _pwd=_pwd, _recer=_recer ,fund_code=None ,fund_name=None):
        self._user = _user
        self._pwd = _pwd
        self._recer = _recer
        self.fund_code = fund_code
        self.fund_name = fund_name

    @staticmethod
    def action_send(self):
        """
        发送邮件
        """
        #创建一个带附件的实例
        msg=MIMEMultipart()
        msg["Subject"] = " don't panic"
        msg["From"] = _user
        msg["To"] = ",".join(_recer)#区别与给一个人发，指定某个人用 msg["To"] = _to 多个人用.join
        #邮件正文内容
        msg.attach(MIMEText('请先想想这一点。','plain', 'utf-8'))
        #构造附件1，传输当前目录下的图片.txt文件
        att1=MIMEText(open('C://temp//upload//test.txt','rb').read(),'base64','utf-8')
        att1['Content-Type']='application/octet-stream'
        att1['Content-Disposition']='attachment;filename="demo.txt"' #filename 填什么，邮件里边展示什么

        #构造附件2，传输当前目录下的图片.jpg文件
        att2=MIMEText(open('C://temp//upload//Figure_1.png','rb').read(),'base64','utf-8')
        att2['Content-Type']='application/octet-stream'
        att2['Content-Disposition']='attachment;filename="Figure_1.png"' #filename填什么，邮件里边展示什么

        #构造附件3，传输当前目录下的图片.jpg文件
        att3=MIMEText(open('C://temp//upload//Figure_2.png','rb').read(),'base64','utf-8')
        att3['Content-Type']='application/octet-stream'
        att3['Content-Disposition']='attachment;filename="Figure_2.png"' #filename填什么，邮件里边展示什么


        #msg.attach(att1)
        msg.attach(att2)
        msg.attach(att3)

        try:
            s=smtplib.SMTP_SSL("smtp.qq.com",465)
            s.login(_user,_pwd)
            s.sendmail(_user,_recer,msg.as_string())
            s.quit()
            print("Success!")
        except smtplib.SMTPException as e:
            print("Failed,%s"%e)



