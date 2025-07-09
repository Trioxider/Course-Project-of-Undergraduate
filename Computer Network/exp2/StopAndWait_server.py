import socket
import random

# 创建UDP套接字
serverPort = 13000
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverSocket.bind(('', serverPort))
print('The server is ready to receive')
# 手动设置ACK的内容
while True:
    # 从client端接收信息和地址
    data, addr = serverSocket.recvfrom(1024)
    packet_num = int(data.split()[1])
    print(f'receive message {data} from {addr}')
    # 用随机数模拟丢包
    rand = random.randint(1, 10)
    if rand <= 4:
        print("Packet lost")
        continue
    else:
        # 未丢包，按正常流程发送ACK给client
        ACK = 'ACK' + str(packet_num)
        serverSocket.sendto(ACK.encode(), addr)
        print(f'ACK is sent to {addr}')
# 关闭套接字链接
serverSocket.close()
