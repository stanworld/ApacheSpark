__author__ = 'stan'

# import socket
# import cPickle as pickle
# client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# address = ('localhost', 9999)
# client.connect(address)
# data=client.recv(1000)
# client.close();
# events=pickle.loads(data)
# print events

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

stream = ssc.socketTextStream("localhost", 9999)

stream.pprint()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate