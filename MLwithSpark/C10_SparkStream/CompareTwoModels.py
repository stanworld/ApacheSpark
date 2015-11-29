__author__ = 'stan'

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from operator import add
import math


# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "Streaming online learning performance comparison")
ssc = StreamingContext(sc, 10)

stream = ssc.socketTextStream("localhost", 9999)

numFeatures = 100

zeroVector=Vectors.zeros(numFeatures)

model1 = StreamingLinearRegressionWithSGD(stepSize=0.01,numIterations=1)
model1.setInitialWeights(Vectors.dense([0]*numFeatures))

model2 = StreamingLinearRegressionWithSGD(stepSize=1,numIterations=1)
model2.setInitialWeights(Vectors.dense([0]*numFeatures))

def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(label=values[0], features=Vectors.dense(values[1:]))

labeledStream=stream.map(lambda line: parsePoint(line))

model1.trainOn(labeledStream)
model2.trainOn(labeledStream)


def calculate(point,latest1,latest2):
    pred1=latest1.predict(point.features)
    pred2=latest2.predict(point.features)
    return (pred1-point.label,pred2-point.label)

def do_operations(rdd):
    latest1=model1.latestModel()
    latest2=model2.latestModel()
    return rdd.map(lambda point: calculate(point,latest1,latest2))

predsAndTrue=labeledStream.transform(lambda rdd: do_operations(rdd))



def operations (time,rdd):
    mse1=rdd.map(lambda (err1,err2): err1*err1).reduce(add)/rdd.count()
    rmse1= math.sqrt(mse1)
    mse2=rdd.map(lambda (err1,err2): err2*err2).reduce(add)/rdd.count()
    rmse2= math.sqrt(mse2)
    print("Batch start time: %s\n" %time)
    print("MSE: Model1 %s, Model2 %s\n" %(mse1,mse2))
    print("RMSE: Model1 %s, Model2 %s\n" %(rmse1,rmse2))

# rdd and time
predsAndTrue.foreachRDD(lambda time,rdd:operations(time,rdd))





#model.predictOnValues(labeledStream.map(lambda lp: (lp.label, lp.features))).pprint()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate


