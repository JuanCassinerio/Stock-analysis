import smtplib
from email.mime.text import MIMEText

subject = "Segui asi"
body = "No seas Boludo... ChotoCorta"
sender = "juancassinerio@gmail.com"
recipients = ["juancassinerio@gmail.com"]
password = "oregonpercei58"
app_password = "mohs orzv qzcf uadx"

def send_email(subject, body, sender, recipients, app_password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
       smtp_server.login(sender, app_password)
       smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Message sent to",recipients)


# Call the function with the updated parameters
send_email(subject, body, sender, recipients, app_password)