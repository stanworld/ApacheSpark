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


