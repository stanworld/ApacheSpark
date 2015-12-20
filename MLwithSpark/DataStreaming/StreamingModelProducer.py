__author__ = 'stan'

import csv
import random
import time
import SocketServer
import threading
import numpy as np


maxEvents = 100
numFeatures = 100
mu=0
sigma=1

w= np.random.normal(mu, sigma, numFeatures)

intercept=random.gauss(mu, sigma)

def oneSample (i):
        x = np.random.normal(mu, sigma, numFeatures)
        y = np.dot(w,x)
        xstr=map(lambda i: str(i),x.tolist())
        noisy = y+intercept
        return ','.join((str(noisy),','.join(xstr)))
 #       return '\t'.join((str(noisy),','.join(x.tolist())))

def generateNoisyData ( n ):
        events=map(lambda i: oneSample(i),range(0,n))
        return  events


class ThreadedTCPRequestHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        cur_thread = threading.current_thread()
        print("Got client connected from: %s" %cur_thread.name)
        while 1:
            time.sleep(2)
            events=generateNoisyData(random.randint(1,maxEvents))

            print("events:\n%s" %events)
#            data = pickle.dumps(events)

            data= '\n'.join(events)
            data= data+'\n'
            print("data: \n%s" %data)
            self.request.sendall(data)



class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass



if __name__ == "__main__":


    HOST, PORT = "localhost", 9999

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    ip, port = server.server_address

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    print ("Socket server started at ip %s with port %s\n" % (ip,port) )
    server.serve_forever()
