import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
from os.path import exists
from skimage.transform import resize


def image_reader(x):
    if exists(f"photos2/{x['ISBN']}.jpg"):
        x["Image"] = "Not Null"
    return x


class ImagesDataset(Dataset):
    def __init__(self):
        books = pd.read_csv("./data/Books.csv", low_memory=False)
        ratings = pd.read_csv("./data/Ratings.csv")
        users = pd.read_csv("./data/Users.csv").sample(10000, random_state=7)

        ratings = ratings.loc[ratings["User-ID"].isin(users["User-ID"].unique())]
        books = books.loc[books["ISBN"].isin(ratings["ISBN"].unique())]
        books = books[books.columns[:5]]

        books["Image"] = np.nan

        books = books.apply(image_reader, axis=1)
        na_image = books.loc[books["Image"].isna(), "ISBN"]
        books.dropna(inplace=True)
        ratings = ratings.loc[~ratings["ISBN"].isin(na_image)]

        self.dataset = ratings.merge(books, on="ISBN")
        self.dataset.reset_index(inplace=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img_path = f"photos2/{self.dataset.loc[item, 'ISBN']}.jpg"
        image = io.imread(img_path)

        return torch.Tensor(resize(image, (3, 224, 224))), self.dataset.loc[item, "Book-Rating"]