import socket

# 创建TCP套接字
host = 'localhost'
serverPort = 12000
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverSocket.bind(('', serverPort))
serverSocket.listen(1)

while True:
    # 从client段获取请求内容和地址
    client_socket, addr = serverSocket.accept()
    print(f"Connect by {addr}")
    # 获取文件名并解码
    file_name = client_socket.recv(1024)
    file_name = file_name.decode()
    try:  # 尝试打开文件
        with open(file_name, 'rb') as file:
            # 获取文件内容
            content = file.read()
            content = content.decode()
            file.close()
            # 发送请求成功的响应和文件
            client_socket.send(b'OK')
            client_socket.send(content.encode())
    except FileNotFoundError:  # 若文件不存在
        # 发送请求失败的响应
        client_socket.send(b'ERROR')
        client_socket.send('File Not Found')
    # 关闭套接字
    client_socket.close()
