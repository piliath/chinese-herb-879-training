# -*- coding: utf-8 -*-
"""163 邮箱通知: 训练进度/完成/异常时发邮件(支持附图片图表)
用法: python notify_email.py ["主题"] ["正文"] [图片路径可选]
"""
import os
import sys
import smtplib
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

SMTP_HOST = 'smtp.163.com'
SMTP_PORT = 465
SENDER = 'lihuahua_ai_bot@163.com'
AUTH_CODE = 'ZQuvJ33k6W4xwv87'
RECEIVER = 'lihuahua_ai_bot@163.com'


def send(subject, body, image_path=None):
    if image_path and os.path.exists(image_path):
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment',
                       filename=os.path.basename(image_path))
        msg.attach(img)
    else:
        msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = formataddr((str(Header('zhong-yao-train-bot', 'utf-8')), SENDER))
    msg['To'] = RECEIVER
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.login(SENDER, AUTH_CODE)
            s.sendmail(SENDER, [RECEIVER], msg.as_string())
        print('email sent OK')
        return True
    except Exception as e:
        print('email send FAILED:', e)
        return False


if __name__ == '__main__':
    subject = sys.argv[1] if len(sys.argv) > 1 else '中草药消融训练: 测试邮件'
    body = sys.argv[2] if len(sys.argv) > 2 else '测试邮件正文。'
    img = sys.argv[3] if len(sys.argv) > 3 else None
    send(subject, body, img)
