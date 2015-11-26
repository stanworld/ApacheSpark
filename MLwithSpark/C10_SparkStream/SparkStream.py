__author__ = 'stan'

import socket
import cPickle as pickle
client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
address = ('localhost', 9999)
client.connect(address)
data=client.recv(1000)
client.close();
events=pickle.loads(data)
print events