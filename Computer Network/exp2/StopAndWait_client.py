import socket
import time

serverName = 'localhost'
serverPort = 13000
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
clientSocket.settimeout(1)
ACK = 'ACK'
count = 0
sum_time = 0
avr_rtt = 0
loss_rate = 0
loss = 0
rtt = 0
packet = int(input("Please input the number of packet"))
while count < packet:
    data = f'segment {count + 1}'
    start_time = time.time()
    clientSocket.sendto(data.encode(), (serverName, serverPort))
    print(f'Send segment {count + 1} to server')
    try:
        ack, addr = clientSocket.recvfrom(1024)
        end_time = time.time()
        rtt = end_time - start_time
        sum_time += rtt
        print(f"Receive {ack} form {addr}")
        count += 1
    except socket.timeout:
        loss += 1
        print("Packet lost")
        clientSocket.sendto(data.encode(), (serverName, serverPort))

clientSocket.close()

loss_rate = loss / packet
if packet == 10:
    print('The loss rate is set to be 0.5 ideally')
elif packet == 100:
    print('The loss rate is set to be 0.2 ideally')
avr_rtt = sum_time/packet
loss_rate = round(loss_rate, 2)
avr_rtt = round(avr_rtt, 4)
sum_time = round(sum_time, 4)
print(f'The real loss rate is {loss_rate}')
print(f'The average RTT is {avr_rtt}')
print(f'The total RTT is {sum_time}')
