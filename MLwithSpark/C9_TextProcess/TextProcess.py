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


print(','.join(whiteSpaceSplit.sample(True,0.3,42).take(100)))


#####splitting each document on nonword characters using regular expression

import re

nonWordSplit=text.flatMap(lambda line: map(lambda x: x.lower(),re.split('\W+',line)))
print(nonWordSplit.distinct().count())

###### remove words containing number digits
pattern='\w*\d\w*'

filterNumbers=nonWordSplit.filter(lambda token: re.search(pattern,token) is None)

print(filterNumbers.distinct().count())

print(",".join(filterNumbers.distinct().sample(True,0.3,42).take(10)))


#####Removing stop words


tokenCounts = filterNumbers.map(lambda t: (t,1)).reduceByKey(add).sortBy(lambda t: t[1], ascending=False)

print(tokenCounts.take(20))


from sets import Set

stopWords = Set(["the","a","an","of","or","in","for","by","on","but","is","not","with","as","was","if","they","are","this","and","it","have","from","at","my","be","that","to",""])


tokenCountsFilteredStopWords = tokenCounts.filter(lambda (k,v): not (k in stopWords)).sortBy(lambda t: t[1], ascending=False)

## remove single letter
tokenCountsFilteredSize=tokenCountsFilteredStopWords.filter(lambda (k,v): len(k)>=2).sortBy(lambda t: t[1], ascending=False)

#### remove tokens that apprear only once
rareTokens = set(tokenCounts.filter(lambda (k,v): v <2).map(lambda(k,v): k).collect())

tokenCountsFilteredAll=tokenCountsFilteredSize.filter(lambda (k,v): not(k in rareTokens)).sortBy(lambda t: t[1], ascending=False)

print(tokenCountsFilteredAll.count())


def tokenize(doc):
    import re
    pattern='\w*\d\w*'
    nonWordSplitl=map(lambda x: x.lower(),re.split('\W+',doc))
    filterNumbersl=filter(lambda token: re.search(pattern,token) is None, nonWordSplitl)
    next1=filter(lambda token: not (token in stopWords), filterNumbersl)
    next2=filter(lambda token: not (token in rareTokens),next1)
    next3=filter(lambda token: len(token)>=2, next2)
    return next3

print(text.flatMap(lambda doc: tokenize(doc)).distinct().count())


##### End of tokenization, note there are no steps taken for stemming.





#Train a TF-IDF model

tokens=text.map(lambda doc: tokenize(doc))

from pyspark.mllib.feature import HashingTF

from pyspark.mllib.feature import IDF

from pyspark.mllib.linalg import SparseVector as SV

dim=pow(2,18)

hashingTF = HashingTF(dim)

tf=hashingTF.transform(tokens)

tf.cache()

v=tf.first()

print(v.size)
print(v.values)
print(v.indices)

idf = IDF().fit(tf)

tfidf=idf.transform(tf)

v2=tfidf.first()

print(v2.size)
print(v2.values)
print(v2.indices)

minMaxVals = tfidf.map(lambda v: (min(v.values),max(v.values)))
globalMin=minMaxVals.reduce(min)
globalMax=minMaxVals.reduce(max)
globalMinMax=(globalMin[0],globalMax[1])

###Using a TF-IDF model

hockeyText= rdd.filter(lambda (file,text): file.find("hockey")!= -1)

hockeyTF=hockeyText.mapValues(lambda doc: hashingTF.transform(tokenize(doc)))

hockeyTfIdf=idf.transform(hockeyTF.map(lambda x: x[1]))

hockey1=hockeyTfIdf.sample(True,0.1,42).first()

hockey2=hockeyTfIdf.sample(True,0.1,43).first()

cosineSim=hockey1.dot(hockey2)/(hockey1.norm(2)*hockey2.norm(2))

### Training a text classifier using TF-IDF

from pyspark.mllib.classification import NaiveBayes

from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.evaluation import MulticlassMetrics


newsgroupsMap=newsgroups.distinct().zipWithIndex().collectAsMap()

zipped = newsgroups.zip(tfidf)

train=zipped.map(lambda (topic,vector): LabeledPoint(newsgroupsMap[topic],vector))

model=NaiveBayes.train(train,0.1)

testPath = "/home/stan/spark_code/MLwithSpark/C9_TextProcess/20news-bydate-test/*"

testRDD = sc.wholeTextFiles(testPath)

testnewsgroups=testRDD.map(lambda (file,text): file.split("/")[-2])

testLabels = testnewsgroups.map(lambda x:newsgroupsMap[x])


testTf = testRDD.map(lambda (file,text): hashingTF.transform(tokenize(text)))

testTfIdf= idf.transform(testTf)

zippedTest = testLabels.zip(testTfIdf)

test = zippedTest.map(lambda (topic,vector): LabeledPoint(topic,vector))

predictionAndLabel = test.map(lambda x: (model.predict(x.features),x.label))

accuracy = 1.0*predictionAndLabel.filter(lambda x: x[0]==x[1]).count()/test.count()

metrics= MulticlassMetrics(predictionAndLabel)

print (accuracy)

print (metrics.weightedFMeasure())


#raw features

rawTokens = rdd.map(lambda (file,text): text.split(" "))

rawTF=rawTokens.map(lambda doc: hashingTF.transform(doc))


rawTrain=newsgroups.zip(rawTF).map(lambda (topic,vector): LabeledPoint(newsgroupsMap(topic),vector))

rawModel = NaiveBayes.train(rawTrain,0.1)

# Word2Vec models

from pyspark.mllib.feature import Word2Vec

word2vec= Word2Vec()

word2vec.setSeed(42)

word2vecModel = word2vec.fit(tokens)

word2vecModel.findSynonyms("hockey",20)

