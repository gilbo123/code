#!/usr/bin/env python

import socket
import time

#TCP_IP_1 = '10.0.0.177'
TCP_IP_2 = '10.0.0.178'
# TCP_PORT_1 = 8001
TCP_PORT_2 = 8002
BUFFER_SIZE = 16#1024
MESSAGE1 = "Hello from Controllino-1!"
MESSAGE2 = ""


while(1):
    '''
    message to first controllino
    '''
    # s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s1.connect((TCP_IP_1, TCP_PORT_1))
    # s1.send(MESSAGE1)
    # #wait for data
    # time.sleep(0.1)
    # data = s1.recv(BUFFER_SIZE)
    # print data
    # s1.close()

    #small sleep
    # time.sleep(0.5)

    '''
    message to second controllino
    '''
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.connect((TCP_IP_2, TCP_PORT_2))
    s2.send(MESSAGE2)
    #wait for data
    time.sleep(0.1)
    data = s2.recv(BUFFER_SIZE)
    print (data)#data.decode("hex")
    s2.close()

    #loop sleep
    time.sleep(.2)

# s.close()
