import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from random import random

h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'

# import the feature vector and trained labels
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features = np.array(h5f_data['dataset_1'])
global_labels = np.array(h5f_label['dataset_1'])

h5f_data.close()
h5f_label.close()

class_names = ["bulbasaur", "charmander", "mewtwo", "pikachu", "squirtle"]

blue_color = '#0000FF'
green_color = '#00FF00'
red_color = '#FF0000'
pink_color = '#FF00FF'
yellow_color = '#FFFF00'

colors = [green_color, red_color, pink_color, yellow_color, blue_color]

ss = StandardScaler()
scaled_features = ss.fit_transform(global_features)

tsne = TSNE(n_components=2, perplexity=50)
tsne_data = tsne.fit_transform(scaled_features)

plt.figure(figsize=(8, 7))

for index, label in enumerate(class_names):
    color = colors[index]
    plt.scatter(tsne_data[global_labels == index, 0], tsne_data[global_labels == index, 1], color=color, label=label)

plt.legend()
plt.show()
