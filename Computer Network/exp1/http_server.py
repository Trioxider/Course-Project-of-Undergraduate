import socket

# 创建socket套接字
serverPort = 12000
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverSocket.bind(('localhost', serverPort))
serverSocket.listen(1)
print(f"Server is listening on localhost:{serverPort}")

while True:
    # 获取请求信息和地址
    conn, addr = serverSocket.accept()
    print(f'Accepted connection from{addr}')
    # 将请求的信息解码
    request = conn.recv(2048).decode()
    print(f'Receive request:{request}')
    # 获取request line
    request_line = request.split("\n")[0]
    print(f'Request line is {request_line}')
    # 从request line 中获取文件名
    file_name = request_line.split()[1]
    print(f'Request file name is {file_name}')

    try:
        # 尝试打开并读取html文件的北荣
        with open(file_name[1:], "rb") as file:
            content = file.read()
            file.close()
        # 编写返回的响应头
        status_line = 'HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n'
        # 将响应头和文件内容组合
        response = status_line .encode() + content
        # 发送相应
        conn.send(response)
        print('Response success: 200')

    except FileNotFoundError:  # 捕获文件打开错误
        # 打开提前建立的404 Error网站
        with open("404.html", "rb") as file:
            content = file.read()
            file.close()
        # 编写响应头
        status_line = 'HTTP/1.1 404 Not Found\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n'
        print('Response failure: 404 Not Found')
        # 组合并发送响应头
        response = status_line.encode() + content
        conn.send(response)

    conn.close()
