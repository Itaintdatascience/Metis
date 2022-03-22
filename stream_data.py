# Stream new data into reviews collection in MongoDB
from schedule import every, repeat, run_pending
from pymongo import MongoClient
import time

client = MongoClient()

db = client.yelpdb
collection_prod = db.prod
# collection_reviews = db_yelp.reviews
# create new collection for gathering data



@repeat(every(60).seconds)
def job():
    """
    Create MongoDB scheduler to stream data from `reviews` to `production` level data
    """
    print ('streamed 5000 docs from `reviews` to `prod` collection')
    docs = list(db.reviews.aggregate([{'$sample': {'size': 5000}}]))
    db.prod.insert(docs)
    db.reviews.delete_many(docs)


while True:
    run_pending()
    time.sleep(1)