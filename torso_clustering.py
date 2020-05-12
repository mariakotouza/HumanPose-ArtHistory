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

#sns.set(rc={'figure.figsize':(11.7,8.27)})
#palette = sns.color_palette("bright", 3)

relative_path = ''
directory = relative_path + 'Data_csv/'
output_dir = relative_path + 'output/'
datasets = ["Caravaggio2", "Caravaggio1", "Raphael", "Rubens"]

n_digits = 4 #len(np.unique(data['dataset'].values))

#keypoints_names_dict = pickle.load( open( relative_path + "Data_csv/" + "keypoints_names_dict.p", "rb" ) )

# Read the data from all datasets and create a dataframe
for dat in datasets:
    if (dat == datasets[0]):
        data = pd.read_csv((directory + dat + "_angles_rotated_with_names.txt"), sep = "\t", index_col=0)
        data['dataset'] = dat
    else:
        temp = pd.read_csv((directory + dat + "_angles_rotated_with_names.txt"), sep="\t", index_col=0)
        temp['dataset'] = dat
        data = data.append(temp, ignore_index=True)

keys_attr = ['dataset', 'human', 'image']
extra_attr = ['rotation', 'rotated']
points_attr_ids = list(range(0,(len(data.columns)-len(keys_attr) - len(extra_attr))))
points_attr = list(data.columns[points_attr_ids])
col_names = keys_attr + points_attr + extra_attr
#print(col_names)

data = data[col_names] # columns: ['dataset', 'human', 'image', 'Unnamed: 0', '1:Neck_0:Nose', .. , '19:LBigToe_14:LAnkle', 'rotation', 'rotated']
#print("Data columns:")
#print(data.columns)

data['image'] = [s.replace(".json", "") for s in data['image']]
images_temp = ['/img/' + data['image'][i] for i in range(len(data['image']))]
dataset_temp = ['../../../Data/' + data['dataset'][i] for i in range(len(data['dataset']))]
images = [m+n for m,n in zip(dataset_temp,images_temp)]

data_all = data

data_cos = np.cos(data['rotation'])
data_sin = np.sin(data['rotation'])

frame = {'torso_sin': data_sin, 'torso_cos': data_cos}
torso_sin_cos = pd.DataFrame(frame)
torso_sin_cos = torso_sin_cos.values

#data_all.to_csv('../../../Data_csv/' + 'data_all' + '_angles_rotated_with_names.txt', sep='\t')

# #############################################################################
# Data for clustering
np.random.seed(42)

labels = data['dataset']

#print(data.shape)
n_samples, n_features = data.shape

# create palette by dataset name
d = dict([ (y1,x1+1) for x1,y1 in enumerate(datasets) ])
colors_scatter = [d[x] for x in labels]

# Torso
torso = data['rotation']
torso = torso.values
#torso = scale(torso)
torso = torso.reshape(-1,1)

# Torso clustering
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(torso_sin_cos)
clusters = kmeans.predict(torso_sin_cos)

###################################################################
# Scatter plot
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = torso_sin_cos[:, 0].min() - 1, torso_sin_cos[:, 0].max() + 1
y_min, y_max = torso_sin_cos[:, 1].min() - 1, torso_sin_cos[:, 1].max() + 1
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

scatter = plt.scatter(torso_sin_cos[:, 0], torso_sin_cos[:, 1], s=5, c=colors_scatter)
plt.legend(handles=scatter.legend_elements()[0], labels=datasets)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering torso angles sin and cos\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.xlabel('sin(x)')
plt.ylabel('cos(x)')

plt.savefig(output_dir + 'torso_clustering.png')
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

X = torso_sin_cos
visualize2DData (X, fig, ax, image_path, centroids, colors_scatter, datasets)

plt.title('K-means clustering torso angles sin and cos\n'
          'Centroids are marked with white cross')
plt.xlabel('sin(x)')
plt.ylabel('cos(x)')
plt.show()

######################################################
# Histogram per cluster id
legend = datasets
cluster_list = []

for dat in datasets:
    ids = [i for i in range(len(labels)) if labels[i] == dat]
    cluster_list.append(list(clusters[ids]))

plt.hist(cluster_list)
plt.xlabel("Cluster id")
plt.ylabel("Frequency")
plt.legend(legend)
plt.xticks(list(set(clusters)))
#plt.yticks(range(1, 20))
plt.title('Histogram of the torso rotation clustering results (per cluster)')

plt.savefig(output_dir + 'histogram_torso_per_cluster.png')
plt.show()

####################################################################
# Histogram per cluster id
legend = ['Cluster ' + str(i) for i in list(set(clusters))]
dataset_list = []

for dat in list(set(clusters)):
    ids = [i for i in range(len(clusters)) if clusters[i] == dat]
    dataset_list.append(list(labels[ids]))

plt.hist(dataset_list)
plt.xlabel("Dataset")
plt.ylabel("Frequency")
plt.legend(legend)
plt.xticks(datasets)
#plt.yticks(range(1, 20))
plt.title('Histogram of the torso rotation clustering results (per artist)')

plt.savefig(output_dir + 'histogram_torso_per_artist.png')
plt.show()
