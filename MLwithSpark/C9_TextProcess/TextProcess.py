__author__ = 'stan'
# shell commands in pyspark
path = "/home/stan/spark_code/MLwithSpark/C9_TextProcess/20news-bydate-train/*"


rdd = sc.wholeTextFiles(path)

text = rdd.map(lambda (file,text): text)

print(text.count())

