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
ssc = StreamingContext(sc, 10)

stream = ssc.socketTextStream("localhost", 9999)

#stream.pprint()

events=stream.map(lambda line: line.split(",")).map(lambda fields: (fields[0],fields[1],fields[2]))

# events is transformedDstream, not RDD
events.pprint()

def operations (time,rdd):
    numPurchases=rdd.count()
    uniqueUsers = rdd.map(lambda (user,product,price):user).distinct().count()
    totalRevenue=rdd.map(lambda (user,product,price):float(price)).sum()
    productByPopularity=rdd.map(lambda(user,product,price):(product,1)).reduceByKey(lambda a,b: a+b).collect()
    mostPopular = sorted(productByPopularity, key=lambda x: x[1], reverse=True)
    print("Batch start time: %s\n" %time)
    print("Total purchases: %s\n" %numPurchases)
    print("Unique users: %s\n" %uniqueUsers)
    print("Total revenues: %s\n" %totalRevenue)
    print(mostPopular)
    print(mostPopular[0])
  #  print("Most popular: %s\n" %mostPopular)
# rdd and time
events.foreachRDD(lambda time,rdd:operations(time,rdd))


ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate