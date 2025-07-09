import socket

# 创建TCP套接字端口
serverName = 'localhost'
serverPort = 12000
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.connect((serverName, serverPort))

# 输入文件名并向server端发送文件请求
file_name = input("请输入文件名：")
file_name = file_name.encode()
clientSocket.send(file_name)
# 从server端接受响应
response = clientSocket.recv(2048)
response = response.decode()

# 如果接收到响应成功的信息
# 创建一个新文件用于储存请求到的信息的内容
if response == 'OK':
    print("File request success——OK")
    file = open('file_recv', 'wb')
    content = clientSocket.recv(2048)
    file.write(content)
    file.close()
# 请求失败则打印ERROR信息
else:
    print("File request failure  ERROR")

