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

##tokenization

whiteSpaceSplit = text.flatMap(lambda line: map(lambda x: x.lower(),line.split(" ")))

print(whiteSpaceSplit.distinct().count())


print(','.join(whiteSpaceSplit.sample(True,0,3,42).take(100)))


#####splitting each document on nonword characters using regular expression

import re

nonWordSplit=text.flatMap(lambda line: map(lambda x: x.lower(),re.split('\W+',line)))
print(nonWordSplit.distinct().count())


pattern='\w*\d\w*'

filterNumbers=nonWordSplit.filter(lambda token: re.search(pattern,token) is None)

print(filterNumbers.distinct().count())

print(",".join(filterNumbers.distinct().sample(True,0.3,42).take(10)))


#####Removing stop words








