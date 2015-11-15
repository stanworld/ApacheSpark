__author__ = 'stan'

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from numpy import array
from pyspark.mllib.recommendation import Rating
from numpy import linalg as LA
import numpy as np

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

print("Exercise done!\n")