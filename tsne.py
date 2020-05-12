import pandas as pd
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns

from Visualization import visualize2DData
from Visualization import visualize3DData

#sns.set(rc={'figure.figsize':(11.7,8.27)})
#palette = sns.color_palette("bright", 3)

replace_na = -1.5
relative_path = ''
directory = relative_path + 'Data_csv/'
output_dir = relative_path + 'output/'
datasets = ["Caravaggio2", "Caravaggio1", "Raphael", "Rubens"]

n_digits = 4 #len(np.unique(data['dataset'].values))

#keypoints_names_dict = pickle.load( open( relative_path + "Data_csv/" + "keypoints_names_dict.p", "rb" ) )

# Read the data from all datasets and create a dataframe
for dat in datasets:
    if (dat == datasets[0]):
        data = pd.read_csv((directory + dat + "_angles_sin_cos.txt"), sep = "\t", index_col=0)
        data['dataset'] = dat
    else:
        temp = pd.read_csv((directory + dat + "_angles_sin_cos.txt"), sep="\t", index_col=0)
        temp['dataset'] = dat
        data = data.append(temp, ignore_index=True)

keys_attr = ['dataset', 'human', 'image']
extra_attr = ['rotation_sin', 'rotation_cos', 'rotated']
points_attr_ids = list(range(0,(len(data.columns)-len(keys_attr) - len(extra_attr))))
points_attr = list(data.columns[points_attr_ids])
col_names = keys_attr + points_attr + extra_attr
#print(col_names)

data = data[col_names] # columns: ['dataset', 'human', 'image', '1:Neck_0:Nose', .. , '19:LBigToe_14:LAnkle', 'rotation', 'rotated']
#print("Data columns:")
#print(data.columns)

data['image'] = [s.replace(".json", "") for s in data['image']]
images_temp = ['/img/' + data['image'][i] for i in range(len(data['image']))]
dataset_temp = ['../../../Data/' + data['dataset'][i] for i in range(len(data['dataset']))]
images = [m+n for m,n in zip(dataset_temp,images_temp)]

data_all = data

data_all.to_csv(relative_path + 'Data_csv/' + 'data_all' + '_angles_rotated_with_names_sin_cos.txt', sep='\t')

# #############################################################################
# Data for clustering
np.random.seed(42)

labels = data['dataset']

#print(data[data.columns[len(data.columns)-1] ])
data = data[points_attr + extra_attr]
data = data.fillna(replace_na)

data = data.values
#data = scale(data)

#print("data used for clustering !!!")
#print(data)

#print(data.shape)
n_samples, n_features = data.shape

# create palette by dataset name
d = dict([ (y1,x1+1) for x1,y1 in enumerate(datasets) ])
colors_scatter = [d[x] for x in labels]

################################################################################
# 2D tsne scatter plot with points_attr + rotated
X = data_all[points_attr + ['rotated']]
X = X.fillna(replace_na)
X = X.values
#X = scale(X)

tsne = TSNE(n_components = 2)
X_embedded = tsne.fit_transform(X)

reduced_data = X_embedded
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

fig = plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c=colors_scatter)
plt.legend(handles=scatter.legend_elements()[0], labels=datasets)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering (tsne-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.savefig(output_dir + 'tsne_2d_kmeas_without_rotation_attr.png')
plt.close(fig)

# #############################################################################
# Data x, y for scatter and an array of images.
# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

image_path = images

X = reduced_data
visualize2DData (X, fig, ax, image_path, centroids, colors_scatter, datasets)
plt.title('K-means clustering (tsne-reduced data)\n'
          'Centroids are marked with white cross')
plt.show()

################################################################################
# 2D tsne scatter plot with points_attr + rotated + rotation
X = data
tsne = TSNE(n_components = 2)
X_embedded = tsne.fit_transform(X)

reduced_data = X_embedded
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig = plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c=colors_scatter)
plt.legend(handles=scatter.legend_elements()[0], labels=datasets)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering (tsne-reduced data) with rotation attr\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.savefig(output_dir + 'tsne_2d_kmeas_with_rotation_attr.png')
plt.close(fig)

# #############################################################################
# Data x, y for scatter and an array of images.
# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

image_path = images

X = reduced_data
visualize2DData (X, fig, ax, image_path, centroids, colors_scatter, datasets)

plt.title('K-means clustering (tsne-reduced data) with rotation attr\n'
          'Centroids are marked with white cross')
plt.show()

##############################################################
# 3D scatter plot without rotation attr
X = data_all[points_attr + ['rotated']]
X = X.fillna(replace_na)
X = X.values
#X = scale(X)

tsne = TSNE(n_components = 3)
X_embedded = tsne.fit_transform(X)

reduced_data3 = X_embedded
# Initializing KMeans
kmeans = KMeans(n_clusters=4)
# Fitting with inputs
kmeans = kmeans.fit(reduced_data3)
# Predicting the clusters
labels_pr = kmeans.predict(reduced_data3)
# Getting the cluster centers
C = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)
scatter = ax.scatter(reduced_data3[:, 0], reduced_data3[:, 1], reduced_data3[:, 2], c=colors_scatter)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

plt.title('K-means clustering (tsne-reduced data) 3D\n'
          'Centroids are marked with white cross')
plt.legend(handles=scatter.legend_elements()[0], labels=datasets, loc=2)

plt.savefig(output_dir + 'tsne_3d_kmeas_without_rotation_attr.png')
plt.close(fig)

################################################################
# 3D plot with images
image_path = images
# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

X = reduced_data3
visualize3DData (X, fig, ax, image_path, C, colors_scatter, datasets)

plt.title('K-means clustering (tsne-reduced data) 3D\n'
          'Centroids are marked with white cross')
plt.show()

##############################################################
# 3D scatter plot with rotation attr
X = data
tsne = TSNE(n_components = 3)
X_embedded = tsne.fit_transform(X)

reduced_data3 = X_embedded
# Initializing KMeans
kmeans = KMeans(n_clusters=4)
# Fitting with inputs
kmeans = kmeans.fit(reduced_data3)
# Predicting the clusters
labels_pr = kmeans.predict(reduced_data3)
# Getting the cluster centers
C = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)
scatter = ax.scatter(reduced_data3[:, 0], reduced_data3[:, 1], reduced_data3[:, 2], c=colors_scatter)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

plt.title('K-means clustering (tsne-reduced data) 3D with rotation attr\n'
          'Centroids are marked with white cross')
plt.legend(handles=scatter.legend_elements()[0], labels=datasets, loc=2)

plt.savefig(output_dir + 'tsne_3d_kmeas_with_rotation_attr.png')
plt.close(fig)

################################################################
# 3D plot with images with rotation attr
image_path = images
# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

X = reduced_data3
visualize3DData (X, fig, ax, image_path, C, colors_scatter, datasets)

plt.title('K-means clustering (tsne-reduced data) 3D with rotation attr\n'
          'Centroids are marked with white cross')
plt.show()