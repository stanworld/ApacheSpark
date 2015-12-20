__author__ = 'stan'

from pyspark import SparkContext
from pyspark.streaming import StreamingContext


# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "statefulstreaming")
ssc = StreamingContext(sc, 10)

ssc.checkpoint("/tmp/sparkstreaming/")

stream = ssc.socketTextStream("localhost", 9999)

def updateState(prices,currentTotal):
    currentRevenue=sum(map(lambda (product,price):float(price),prices))
    currentNumberPurchase=len(prices)
    state=currentTotal or (0,0.0)
    return (currentNumberPurchase+state[0],currentRevenue+state[1])

events=stream.map(lambda line: line.split(",")).map(lambda fields: (fields[0],fields[1],fields[2]))

users=events.map(lambda (user,product,price): (user,(product,price)))

revenuePerUser=users.updateStateByKey(updateState)

revenuePerUser.pprint()


ssc.start()

ssc.awaitTermination()
