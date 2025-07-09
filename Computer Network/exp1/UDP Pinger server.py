import socket
import random

port = 12000
# 创建一个UDP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 绑定本地IP地址和端口号，IP地址为空表示任意地址
server_socket.bind(('', port))
print("Server is already")

# 建立响应循环
while True:
    # 用生成随机数模拟丢包机制
    rand = random.randint(0, 10)
    # 接受client端的请求信息和地址
    message, addr = server_socket.recvfrom(1024)
    if rand < 4:  # 丢包情况
        # 不做回应直到超时
        continue
    else:
        # 未丢包，正常发送响应报文
        Message = "Pong" + message.decode()
        server_socket.sendto(Message.encode(), addr)
# 关闭套接字
server_socket.close()
