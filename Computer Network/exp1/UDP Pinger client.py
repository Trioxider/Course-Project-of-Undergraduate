import socket
from datetime import datetime
# import statistics

sever_name = "localhost"
Port = 12000
# 创建UDP套接字
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 设置超时时间
clientSocket.settimeout(1)
# 设置发送报文数
messageLen = 10
# 丢包数
loseMessage = 0
# 总时间
sumTime = 0

for i in range(messageLen):
    now = datetime.now()
    print("Ping %d %s" % (i, now))
    # 设置发送报文（0-9的数字共10个）
    message = ('Ping %d %s' % (i, now)).encode()
    try:
        # 获取开始时间
        start_time = datetime.now()
        # 将信息发送到服务器
        clientSocket.sendto(message, (sever_name, Port))
        # 获取server端的信息和地址
        Message, addr = clientSocket.recvfrom(1024)
        # 获取结束时间
        end_time = datetime.now()
        # 计算往返时间
        RTT = (start_time - end_time).total_seconds()
        # 将时间求和
        sumTime += RTT
        print("%s" % Message)

    except socket.timeout:  # 捕获超时异常
        # 记录丢包数
        loseMessage += 1
        print("Request time out")

# 计算丢包率和平均往返时间
percent = loseMessage / messageLen
avr_TTL = sumTime / messageLen
print("The loss packet is %d" % loseMessage)
print("The loss percent is %.2f" % percent)
# 关闭套接字
clientSocket.close()
