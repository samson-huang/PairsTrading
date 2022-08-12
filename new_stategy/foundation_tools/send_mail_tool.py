#!/usr/bin/env python
#-*-coding:utf-8-*-

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import json
import warnings
import datetime
import numpy as np

warnings.filterwarnings('ignore')
# 请根据自己的情况填写ts的token
setting = json.load(open('C://config//config.json'))


_user = "375317196@qq.com"
_pwd = setting['qq']
# _to = "ft_clover@163.com"
_recer=["tuo.huang@zdzq.com.cn","huangtuo02@163.com",]

class send_mail_tool:
    def __init__(self, _user=_user, _pwd=_pwd, _recer=_recer ,fund_code=None ,fund_name=None, local_url=None):
        self._user = _user
        self._pwd = _pwd
        self._recer = _recer
        self.fund_code = fund_code
        self.fund_name = fund_name
        self.local_url = local_url

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
        msg.attach(MIMEText('相关指数代码对应 000009.SH_上证380,000010.SH_上证180, 000016.SH_上证50, 000300.SH_沪深300, 000688.SH_科创50,'+
              '000852.SH_中证1000, 000903.SH_中证100, 000905.SH_中证500, 000906.SH_中证800,/n'+
          '399001.SZ_深证成指,399005.SZ_中小板指, 399006.SZ_创业板指, 399330.SZ_深证100 /n '+
           '图形说明请见《图形说明.png》文件','plain', 'utf-8'))

        #构造附件1，传输当前目录下的图片.txt文件
        att1=MIMEText(open('C://temp//upload//图形说明.gif','rb').read(),'base64','utf-8')
        att1['Content-Type']='application/octet-stream'
        att1['Content-Disposition']='attachment;filename="图形说明.gif"' #filename 填什么，邮件里边展示什么
        msg.attach(att1)

        list_1 = np.load('C://temp//upload//index_list.npy')
        list_1 = list_1.tolist()
        for index_code in list_1:
            #构造附件2，传输当前目录下的图片.jpg文件
            local_url_new=self.local_url+'_'+index_code.replace('.', '')+'_detail.jpg'
            att2=MIMEText(open(local_url_new,'rb').read(),'base64','utf-8')
            att2['Content-Type']='application/octet-stream'
            att2['Content-Disposition']='attachment;filename="'+index_code.replace('.', '')+'.jpg''"' #filename填什么，邮件里边展示什么
            msg.attach(att2)

            local_url_new=self.local_url+'_'+index_code.replace('.', '')+'_overall.jpg'
            att3=MIMEText(open(local_url_new,'rb').read(),'base64','utf-8')
            att3['Content-Type']='application/octet-stream'
            att3['Content-Disposition']='attachment;filename="'+index_code.replace('.', '')+'.jpg''"' #filename填什么，邮件里边展示什么
            msg.attach(att3)
        #msg.attach(att1)

        #msg.attach(att3)

        try:
            s=smtplib.SMTP_SSL("smtp.qq.com",465)
            s.login(self._user,self._pwd)
            s.sendmail(self._user,self._recer,msg.as_string())
            s.quit()
            print("Success!")
        except smtplib.SMTPException as e:
            print("Failed,%s"%e)



