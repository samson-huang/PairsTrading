#!/usr/bin/env python
#-*-coding:utf-8-*-
'''
功能:邮件发送
创建人:
邮箱:
创建日期:2024年4月9日
版本:1.0
'''
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import json
import warnings
import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
# 请根据自己的情况填写ts的token
setting = json.load(open('C://config//config.json'))


_user = "375317196@qq.com"
_pwd = setting['qq']
# _to = "ft_clover@163.com"
_recer=["tuo.huang@zdzq.com.cn","huangtuo02@163.com",]

class send_mail_tool:
    def __init__(self, _user=_user, _pwd=_pwd, _recer=_recer ,fund_code=None ,fund_name=None, local_url=None,name_list=None):
        self._user = _user
        self._pwd = _pwd
        self._recer = _recer
        self.fund_code = fund_code
        self.fund_name = fund_name
        self.local_url = local_url
        self.name_list = name_list

    def action_send(self):
        """
        发送邮件
        """
        local_datetime = datetime.datetime.now().strftime('%Y%m%d')
        #创建一个带附件的实例
        msg=MIMEMultipart()
        msg["Subject"] = local_datetime+'——'+'相关指数波段预测'
        msg["From"] = self._user
        msg["To"] = ",".join(self._recer)#区别与给一个人发，指定某个人用 msg["To"] = _to 多个人用.join


        #邮件正文内容
        msg.attach(MIMEText('相关指数代码对应/n '+
           '图形说明请见《图形说明.png》文件','plain', 'utf-8'))

        #构造附件1，传输当前目录下的图片.txt文件
        att1=MIMEText(open('C://temp//upload//message.gif','rb').read(),'base64','utf-8')
        att1['Content-Type']='application/octet-stream'
        att1['Content-Disposition']='attachment;filename="message.gif"' #filename 填什么，邮件里边展示什么
        msg.attach(att1)

        #list_1 = np.load('C://temp//upload//index_list.npy')
        #list_1 = list_1.tolist()
        list_1 = self.name_list
        #with open('C://temp//upload//codefundsecname.json') as file:
            #code2secname = json.loads(file.read())
        dir_name = 'c:\\temp\\upload\\codefundsecname.csv'
        codefundsecname = pd.read_csv(dir_name)

        for index_code in list_1:
            # 构造附件2，传输当前目录下的图片.jpg文件
            code_name = codefundsecname[codefundsecname['code'] == index_code]['name'].str.strip()
            code_name_new = code_name.astype(str).values[0]
            local_url_new = self.local_url + '_' + code_name_new + '_detail.jpg'
            try:
                with open(local_url_new, 'rb') as f:
                    att2 = MIMEText(f.read(), 'base64', 'utf-8')
                    att2['Content-Type'] = 'application/octet-stream'
                    att2['Content-Disposition'] = 'attachment;filename="' + index_code.replace('.', '') + '.jpg' + '"'
                    msg.attach(att2)
            except FileNotFoundError:
                print(f"File not found: {local_url_new}. Skipping attachment.")

            local_url_new = self.local_url + '_' + code_name_new + '_overall.jpg'
            try:
                with open(local_url_new, 'rb') as f:
                    att3 = MIMEText(f.read(), 'base64', 'utf-8')
                    att3['Content-Type'] = 'application/octet-stream'
                    att3['Content-Disposition'] = 'attachment;filename="' + index_code.replace('.', '') + '.jpg' + '"'
                    msg.attach(att3)
            except FileNotFoundError:
                print(f"File not found: {local_url_new}. Skipping attachment.")

        try:
            s = smtplib.SMTP_SSL("smtp.qq.com", 465)
            s.login(self._user, self._pwd)
            s.sendmail(self._user, self._recer, msg.as_string())
            s.quit()
            print("Success!")
        except smtplib.SMTPException as e:
            print("Failed, %s" % e)



