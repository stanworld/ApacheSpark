__author__ = 'stan'

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from numpy import array
from pyspark.mllib.recommendation import Rating
from numpy import linalg as LA
import numpy as np
from operator import add
import math


from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.evaluation import RankingMetrics

sc = SparkContext("local[2]", "Recommendation")

rawData=sc.textFile("/home/stan/spark_data/ml-100k/u.data")

rawRatings=rawData.map(lambda line : line.split("\t")).map(lambda fields : fields[0:3])

ratings=rawRatings.map(lambda fields: Rating(fields[0],fields[1],fields[2]))

model=ALS.train(ratings,50,10,0.01)

model.predict(789,123) ## the prediction score of item 123 for user 789

topKRecs=model.recommendProducts(789,10) ## recommend top 10 items for user 789 based on predicted score.

movies=sc.textFile("/home/stan/spark_data/ml-100k/u.item")

titles=movies.map(lambda line: line.split("|")).map(lambda fields: (int(fields[0]),fields[1])).collectAsMap() # movie id, title map.

## check the relatios between rated movies and recommended movies

moviesForUserAll=ratings.keyBy(lambda line: int(line.user))

moviesForUser=moviesForUserAll.filter(lambda line: line[0]==789)
# top 10 rated
moviesForUser.sortBy(lambda line: line[1].rating, ascending=False).map(lambda line: (titles[line[1].product], line[1].rating)).take(10)


topKRecs=sc.parallelize(topKRecs) # previous return value is a list, to use it as RDD, this step is required.

# top 10 recommended

def f(x):
    print(x)

topKRecs.map(lambda line: (titles[line.product], line.rating)).foreach(f)


def cosineSimilarity( x, y):
    rr=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return rr

itemId=567
itemfactor=model.productFeatures().lookup(itemId)[0].tolist()
itemfactor = np.array(itemfactor)

f(cosineSimilarity(itemfactor,itemfactor));

def itemSimilarity( id, factor):
    sim=cosineSimilarity(factor, itemfactor)
    return (id, sim)
## similarity scores for all the items when compared to item 567
sims=model.productFeatures().map(lambda(id, factor): itemSimilarity(id, factor.tolist()))

sortedItems = sims.top(10,key=lambda x: x[1])

print(titles[itemId])

sortedItems = sc.parallelize(sortedItems)

sortedItems2=sortedItems.map(lambda x: (titles[x[0]],x[1]))

sortedItems2.collect()

# mean squared error

actualRating=moviesForUser.take(1)[0][1]

predictedRating= model.predict(789, actualRating.product)

squaredError = pow(predictedRating-actualRating.rating,2.0)

usersProducts= ratings.map(lambda x: (x.user,x.product))

predictions=model.predictAll(usersProducts).map( lambda x: ((x.user,x.product),x.rating))

ratingsAndPredictions = ratings.map( lambda x: ((int(x.user),int(x.product)),float(x.rating))).join(predictions)

MSE=ratingsAndPredictions.map(lambda ((x,y),(z,t)) : pow(z-t,2)).reduce(add)/ratingsAndPredictions.count()

print ("Mean Squared Error %s\n" %MSE)

RMSE = math.sqrt(MSE);

print ("Rooted Mean Squared Error %s\n" %RMSE)

# Mean average precision at K for user 789

actualMovies=moviesForUser.map(lambda x: int(x[1].product))
predictedMovies = topKRecs.map(lambda x: x.product)

actual=actualMovies.collect()
predicted = predictedMovies.collect()

def avgPrecisionK( actual, predicted, k):
    score=0.0
    numHit=0
    predKr = np.array(predicted[0:k])

    predK =[]
    for index,x in np.ndenumerate(predKr):
       predK.append((x,index[0]))

    for (a,b) in predK:
        if a in actual:
            numHit=numHit+1
            score = score+ numHit/(float(b)+1.0)

    if not actual:
        return 1.0
    else:
        score=score/float(min(len(actual),k))

    return score

# copy functions to IPYTHON
#1. Copy the lines you want to copy into IPython into the clipboard
#2. Enter %paste into IPython
#3. Press enter
apk10=avgPrecisionK(actual,predicted,10)
print ("Average precision in 10:")
f(apk10)

# average precision for all users

itemFactors = model.productFeatures().map( lambda (id,factor): factor).collect()

itemMatrix = np.matrix(itemFactors)
print("Item factor matrix:")
f(itemMatrix.shape)

imBroadcast=sc.broadcast(itemMatrix)

def multivector (userId, array0):
    userVector= np.array(array0)
    scores = imBroadcast.value.dot(userVector)
    scores = np.array(scores)[0]
    sortedWithId =[]
    for index,x in np.ndenumerate(scores):
       sortedWithId.append((x,index[0]))


    sortedWithId.sort(key=lambda tup: tup[0], reverse=True)
    recommendationIds=map(lambda x: (x[0],x[1]+1),sortedWithId)
    return (userId, recommendationIds)

allRecs = model.userFeatures().map( lambda x: multivector(x[0],x[1]))

userMovies = ratings.map(lambda x: (int(x.user),x.product)).groupBy(lambda x: x[0])
userMovies = userMovies.map(lambda x: (x[0], list(x[1])))

K=10

def fx(predicted, actualWithIds):
    actual = map(lambda x: int(x[1]),actualWithIds)
    return avgPrecisionK(actual,predicted, K)

MAPK=allRecs.join(userMovies).map(lambda (userId, (predicted,actualWithIds)): fx(predicted, actualWithIds)).reduce(add)/allRecs.count()

print("Mean Average Precision at K=10 %s" %MAPK)

## use evalution metrics within mllib

predictedAndTrue=ratingsAndPredictions.map(lambda ((x,y), (p,q)): (p,q))

regressionMetrics = RegressionMetrics(predictedAndTrue)

print ("Mean squared error using mllib: %s" %regressionMetrics.meanSquaredError)

predictedAndTrueForRanking = allRecs.join(userMovies).map(lambda (userId, (predicted,actualWithIds)): (map(lambda x: x[1] ,predicted), map(lambda x: int(x[1]),actualWithIds)))

rankingMetrics= RankingMetrics(predictedAndTrueForRanking)

print("Mean average precision using mllib = %s" %rankingMetrics.meanAveragePrecision)

K=2000

MAPK2000=allRecs.join(userMovies).map(lambda (userId, (predicted,actualWithIds)): fx(predicted, actualWithIds)).reduce(add)/allRecs.count()

print("Mean average precision MAPK2000 = %s" %MAPK2000)


print("Exercise done!\n")