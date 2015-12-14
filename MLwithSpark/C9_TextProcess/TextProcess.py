__author__ = 'stan'
# shell commands in pyspark

# data source: http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz

from operator import add

path = "/home/stan/spark_code/MLwithSpark/C9_TextProcess/20news-bydate-train/*"


rdd = sc.wholeTextFiles(path)

text = rdd.map(lambda (file,text): text)


print(text.count())

newsgroups=rdd.map(lambda (file,text): file.split("/")[-2])

countByGroup = newsgroups.map(lambda n: (n,1)).reduceByKey(add).sortBy(lambda (a,b): b, ascending=False).collect()

print(countByGroup)
