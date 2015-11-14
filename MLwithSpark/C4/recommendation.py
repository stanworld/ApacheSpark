__author__ = 'stan'

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from numpy import array
from pyspark.mllib.recommendation import Rating

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

print("Exercise done!\n")