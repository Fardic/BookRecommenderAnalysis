import pandas as pd
import requests
import urllib.request
from os.path import exists

# Data import and observation
# ----------------------------------------------------------------------------------------------------------------------
pd.set_option('display.max_columns', None)
books = pd.read_csv("./data/Books.csv", low_memory=False)

books.head()
books.columns
books.info()

ratings = pd.read_csv("./data/Ratings.csv")
ratings.head()
ratings.info()

users = pd.read_csv("./data/Users.csv")
users.head()
users.info()

urllib.request.urlretrieve(books.loc[6, "Image-URL-M"], f"photos/{books.loc[6, 'ISBN']}.jpg")

def foo(x):
    if not exists(f"photos/{x['ISBN']}.jpg"):
        try:
            urllib.request.urlretrieve(x["Image-URL-L"], f"photos/{x['ISBN']}.jpg" )
        except:
            print("AHHH!!!")

books.apply(foo, axis=1)


