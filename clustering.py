import pandas as pd
#import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns; #sns.set(style="ticks", color_codes=True)
from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import proj3d

from Visualization import visualize2DData
from Visualization import visualize3DData

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

data = data[col_names] # columns: ['dataset', 'human', 'image', 'Unnamed: 0', '1:Neck_0:Nose', .. , '19:LBigToe_14:LAnkle', 'rotation', 'rotated']
#print("Data columns:")
#print(data.columns)

data['image'] = [s.replace(".json", "") for s in data['image']]
images_temp = ['/img/' + data['image'][i] for i in range(len(data['image']))]
dataset_temp = ['../../../Data/' + data['dataset'][i] for i in range(len(data['dataset']))]
images = [m+n for m,n in zip(dataset_temp,images_temp)]

img_ids = data['human']
images_temp_skel = [s.replace(".jpg", "") for s in images_temp]
images_temp_skel = [s.replace("img", "img_skel") for s in images_temp_skel]
images_temp_skel = [images_temp_skel[i] + "_" + str(img_ids[i]) + '.jpg' for i in range(len(data['dataset']))]
images_skel = [m+n for m,n in zip(dataset_temp,images_temp_skel)]
print(images_skel)
images = images_skel

data_all = data

data_all.to_csv(relative_path + 'Data_csv/' + 'data_all' + '_angles_rotated_with_names.txt', sep='\t')

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

sample_size = 300

# create palette by dataset name
d = dict([ (y1,x1+1) for x1,y1 in enumerate(datasets) ])
colors_scatter = [d[x] for x in labels]

#################################################################
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
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
plt.figure(1)
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
plt.title('K-means clustering (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.savefig(output_dir + 'pca_2d_kmeas.png')
plt.show()

# #############################################################################
# Data x, y for scatter and an array of images
fig = plt.figure()
ax = fig.add_subplot(111)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

image_path = images

X = reduced_data
visualize2DData (X, range(len(image_path)), fig, ax, image_path, centroids, colors_scatter, datasets)

plt.title('K-means clustering (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.show()

# #############################################################################
# Create linear plots - one per image - with different colors based on the image label
Z_pr = kmeans.labels_

from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors =  [k  for  k in  colors.keys()]

#colors = ['r', 'b', 'g', 'c', 'm', 'y']

# create a color palette
palette = plt.get_cmap('Set1')

linear = []
#print(data)
#print(Z_pr)
#print(np.unique(Z_pr))

for x in range(1,data.shape[0]):
    linear = linear + plt.plot(data[x-1,:], color=colors[Z_pr[x-1]])

plt.axis('equal')

#plt.legend(handles=linear, labels=list(map(lambda x : x + 1, np.unique(Z_pr))));
plt.savefig(output_dir + 'linear_allinone_plots_pca_kmeas.png')
plt.show()

# shape from wide to long with melt function in pandas
#data2 = data_all.drop('Unnamed: 0', axis=1)
data2 = data_all
data2['Z_pr'] = Z_pr
data2["id"] = data2["image"] + data2["human"].astype(str)
data2 = data2.fillna(replace_na)

scaler = MinMaxScaler()
#print(data2.columns)
data2[points_attr + extra_attr] = scaler.fit_transform(data2[points_attr + extra_attr])
#print(data2['4dab842f-2648-4a41-bec5-d1ac518228d3_2dd70df6-c4cf-438b-8f3d-2870b9ce686a'] )
data3 = data2

# kmeans using all the attributes, not only the pca components
kmeans_all = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(data2[points_attr + extra_attr].values)
data2['Z_pr'] = kmeans.labels_

df2 = pd.melt(data3, id_vars=['dataset', 'id', 'human','image', 'Z_pr'], var_name='metrics', value_name='values')

#print(df2.columns)
#print(df2['metrics'])
#print(df2['values'])

# map continental to color
use_for_color = [1,2,3, 4] # df2.Z_pr.unique() #df2.dataset.unique() #
colors = {con:color for con, color in zip(use_for_color, ['r','b','g', 'y'])}

# create palette by dataset name
d = dict([(y1,x1+1) for x1,y1 in enumerate(datasets)])
l = [d[x] for x in labels]

pal = {d:colors[con] for d, con in set(zip(df2.id, l))}

g = sns.FacetGrid(df2, col="Z_pr", hue='id', palette=pal, col_wrap=1)
g = g.map(plt.plot, "metrics", "values", marker="o")

plt.tight_layout()
plt.savefig(output_dir + 'linear_plots_pca_kmeas.png', dpi = 400)
plt.show()

################################################################
# inverse keypoints' names
# find the optimal number of clusters

scores = [KMeans(n_clusters=i+2).fit(data2[points_attr + extra_attr] ).inertia_
          for i in range(10)]
sns.lineplot(np.arange(2, 12), scores)
plt.xlabel('Number of clusters')
plt.ylabel("Inertia")
plt.title("Inertia of Cosine k-Means versus number of clusters")
#plt.savefig("intertia_cosine_kmeans.jpg", dpi=300)

plt.savefig(output_dir + 'kmeans_cosine_eval.png')
plt.show()

################################################################
# 3d plot
reduced_data3 = PCA(n_components=3).fit_transform(data)
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

plt.title('K-means clustering (PCA-reduced data) 3D\n'
          'Centroids are marked with white cross')
plt.legend(handles=scatter.legend_elements()[0], labels=datasets, loc=2)

plt.savefig(output_dir + 'pca_3d_kmeas.png')
plt.show()


################################################################
# 3D plot with images
image_path = images
# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

X = reduced_data3
visualize3DData (X, fig, ax, image_path, C, colors_scatter, datasets)

plt.title('K-means clustering (PCA-reduced data) 3D\n'
          'Centroids are marked with white cross')
plt.show()