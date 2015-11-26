__author__ = 'stan'

import csv
import random
import time
import SocketServer
import socket
import threading
import cPickle as pickle
with open('names.csv','rt') as fin:
        cin=csv.reader(fin)
        names= [row for row in cin]



products = [
        ("iPhone cover",9.99),
        ("Headphones", 5.49),
        ("Samsung Galaxy Cover",8.95),
        ("iPad Cover",7.49)
    ]


def oneEvent (i):
         index=random.randint(0,3)
         product=products[index][0]
         price =products[index][1]
         random.shuffle(names[0]);
         user=names[0][0];
         return ','.join((user,product,str(price)))

def generateProductEvents ( n ):
        events=map(lambda i: oneEvent(i),range(0,n))
        return  events


class ThreadedTCPRequestHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        cur_thread = threading.current_thread()
        print("Got client connected from: %s" %cur_thread.name)
        while 1:
            time.sleep(2)
            events=generateProductEvents(random.randint(5,10))

            print("events:\n%s" %events)
#            data = pickle.dumps(events)

            data= '\n'.join(events)
            data= data+'\n'
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
