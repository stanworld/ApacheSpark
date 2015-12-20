__author__ = 'stan'

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "Streaming online learning")
ssc = StreamingContext(sc, 10)

stream = ssc.socketTextStream("localhost", 9999)

numFeatures = 100

zeroVector=Vectors.zeros(numFeatures)

model = StreamingLinearRegressionWithSGD(stepSize=0.01,numIterations=1)
model.setInitialWeights(Vectors.dense([0]*numFeatures))


#labeledStream=stream.map(lambda line: line.split('\t')).map(lambda fields: LabeledPoint(float(fields[0]),fields[1].map(lambda line: line.split(',')).map(float())))

def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(label=values[0], features=Vectors.dense(values[1:]))

labeledStream=stream.map(lambda line: parsePoint(line))

model.trainOn(labeledStream)

model.predictOn(labeledStream.map(lambda lp: lp.features)).pprint()

#model.predictOnValues(labeledStream.map(lambda lp: (lp.label, lp.features))).pprint()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate
