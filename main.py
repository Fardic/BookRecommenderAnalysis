import pandas as pd
import requests
import urllib.request
from os.path import exists
import numpy as np
import cv2
from resnet import CNN
import torch
from customDataset import ImagesDataset
from torch.utils.data import DataLoader
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# Images are encoded with Resnet152
# ----------------------------------------------------------------------------------------------------------------------
# dataset = ImagesDataset()
# dataloader = DataLoader(dataset=dataset, batch_size=64)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cnn = CNN().to(device=device)
#
# results = []
# count = 1
# for images, rankings in dataloader:
#     print(count)
#     cnn.eval()
#     sample_ims = images.cuda()
#     encoded = cnn(sample_ims)
#     count += 1
#     results.append(encoded)
#
# torch.save(results, "./data/results2.pt")


# ----------------------------------------------------------------------------------------------------------------------
# New dataset changed into csv
# results = torch.load("./data/results.pt")
# tensor_of_tensors = torch.stack((results[:-1]))
# tensor_of_tensors = tensor_of_tensors.view(tensor_of_tensors.size(0) * tensor_of_tensors.size(1), tensor_of_tensors.size(2))
# last_tot = torch.cat((tensor_of_tensors, results[-1]), 0)
# last_tot = last_tot.cpu().detach().numpy()
#
# last_tot.shape
# dataset.dataset.head()
# dataset.dataset.drop("Image", axis=1, inplace=True)
#
# new_data = pd.concat([dataset.dataset, pd.DataFrame(last_tot)], axis=1)
# new_data.head()
# new_data.shape
# new_data.drop("index", axis=1, inplace=True)
# new_data.to_csv("./data/encoded_dataset.csv", index=False)

# ----------------------------------------------------------------------------------------------------------------------
# Analysis Begins
dataset = pd.read_csv("./data/encoded_dataset.csv")
x = dataset.drop(columns=["Book-Rating"])
y = dataset["Book-Rating"]
# X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=7)
# x.head()

book_encoded = x.drop_duplicates(subset=["ISBN"])
book_encoded.head()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(5, 15))
elbow.fit(book_encoded[[str(i) for i in range(300)]])
elbow.elbow_value_

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(book_encoded[[str(i) for i in range(300)]])

book_encoded["clusters"] = kmeans.labels_

book_map = book_encoded[["ISBN", "clusters"]]
book_map.set_index("ISBN", inplace=True)

x = x[["User-ID", "ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]]

x["clusters"] = x["ISBN"].map(book_map.to_dict())










