__author__ = 'stan'
from pyspark import SparkContext

rawData = sc.textFile("/home/stan/Downloads/train_noheader.tsv")
