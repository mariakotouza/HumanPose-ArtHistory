import json
import os
import pandas as pd
import math
from math import atan2,degrees
import pickle
import numpy as np

dataset = "Caravaggio2"  # "Caravaggio2", "Caravaggio1", "Raphael", "Rubens"

# Read meta file
relative_path = ''
with open(relative_path + 'Data/meta.json') as f:
  data = json.load(f)

for key, value in data.items():
  #print(type(value))
  for x in value:
    #print(type(x))
    for k, v in x.items():
      #print(k, ":", v)
      #print(type(v))
      if (k == "geometry_config"):
        geometry_config = v

for k, v in geometry_config.items():
  #print(k, ":", v)
  #print(type(v))
  if (k == "edges"):
    edges = v
    #for x in v:
      #print(x)
      #print(type(x))

  if (k == "nodes"):
    nodes = v
    #for k2, v2 in v.items():
      #print(k2, ":", v2)
      #print(type(v2))

# Read data
directory = relative_path + 'Data/' + dataset + "/ann"
img = os.listdir(directory)
df = pd.DataFrame()

for i in img:
  count_hum = 0
  with open(directory + "/" + i) as f:
    data = json.load(f)
    for key, value in data.items():
      for x in value:
        if type(x) is dict:
          if (x['classTitle'] == "Human Skeleton"):
            count_hum = count_hum + 1;
          for k, v in x.items():
            if (k == "nodes"):  # save it to df
              temp = pd.DataFrame(v.items())
              v1 = [y['loc'] for y in v.values() if 'loc' in y]
              temp[1] = v1
              temp["image"] = i
              temp["human"] = count_hum
              df = df.append(temp, ignore_index=True)

df['idx'] = df.groupby('image').cumcount()+1

df = df.pivot_table(index=['image', 'human'], columns=0,
                    values=[1], aggfunc='first')

df = df.sort_index(axis=1, level=1)
df.columns = [f'{y}' for x,y in df.columns]
df = df.reset_index()

#print(df)
#print(df.columns)
df_col_withIds = df.columns

# Match columns to body keypoints
temp = pd.DataFrame(nodes.items())

v1 = [y['label'] for y in nodes.values() if 'label' in y]
temp[1] = v1

temp = temp.set_index(0).T.to_dict('list')
keypoints_names_dict = temp
pickle.dump( temp, open( relative_path + "Data_csv/" + "keypoints_names_dict.p", "wb" ) )

#print(temp)

df.rename(columns=temp, inplace=True)
df.columns = [y[0] for y in df.columns]

#print(df)

ids = [y.split(':')[0] for y in df.columns[2:len(df.columns)]]
ids = list(map(int, ids))
#print(ids)
ids = sorted(range(len(ids)), key=lambda k: ids[k])
ids = [y+2 for y in ids]
l = [0,1]
l = l + ids

df = df[df.columns[l]]
#print(df.columns)

df.to_csv(relative_path + 'Data_csv/' + dataset + '.txt', sep='\t')

# column labels with ids
df.columns = df_col_withIds[l]
df.to_csv(relative_path + 'Data_csv/' + dataset + '_colIds.txt', sep='\t')

#print(df)

# Compute the angles for all edges
def is_nan(x):
  return isinstance(x, float) and math.isnan(x)

def GetAngleOfLineBetweenTwoPoints(p1, p2):
  #print("p1~~~~~~~~~~~~~~p2")
  #print(p1, ",", p2)
  if (is_nan(p1) | is_nan(p2)):
    return float('nan')
  xDiff = p2[0] - p1[0]
  yDiff = p2[1] - p1[1]
  return degrees(atan2(yDiff, xDiff))

# print(edges) # list of dict

df_angles = pd.DataFrame()
names_edges = []

for e in edges:
  temp = pd.concat([df[[e["src"]]], df[[e["dst"]]]], axis=1)
  temp.columns = ["src","dst"]
  t = temp.apply(lambda x: GetAngleOfLineBetweenTwoPoints(x.src, x.dst), axis=1)
  df_angles[e["src"] + '_' + e["dst"]] = t
  #df_angles_with_names[keypoints_names_dict[e["src"]][0] + '_' + keypoints_names_dict[e["dst"]][0]] = t
  names_edges = names_edges + [keypoints_names_dict[e["src"]][0] + '_' + keypoints_names_dict[e["dst"]][0]]

#print(df_angles.columns)
#print(names_edges)

df_angles_rot = df_angles

def f(x):
  if x == x:
    return 1
  else:
    return 0

df_rot_index =  df_angles_rot.apply(lambda x: f(x['8ad1caa8-fefe-42b0-b527-6de26daa60a5_d8bbaaa1-dde0-4324-a906-adb382e9ef90']), axis=1)
#print(df_rot_index)

angle_torso = df_angles["8ad1caa8-fefe-42b0-b527-6de26daa60a5_d8bbaaa1-dde0-4324-a906-adb382e9ef90"]
#print(angle_torso)

def f2(x):
  if x == x:
    return x
  else:
    return -90

# if the human pose has a torso than angle_torso remains as it is, else set angle_torso = -90 so that the final rotation to be equal to 0.
angle_torso =  df_angles_rot.apply(lambda x: f2(x['8ad1caa8-fefe-42b0-b527-6de26daa60a5_d8bbaaa1-dde0-4324-a906-adb382e9ef90']), axis=1)
#print(angle_torso)

def rotate(x):
  diff = -90 - angle_torso
  result = x + diff
  result = result.where(result < 180, result - 360) # if diff > 180 then diff = diff -360
  result = result.where(result > -180, 360 + result)  # if diff < -180 then diff = 360 - diff
  return result

df_angles_rot = df_angles_rot.apply(rotate)
rotation = -90 - angle_torso

df_angles['image'] = df['image']
df_angles['human'] = df['human']

df_angles_rot['image'] = df['image']
df_angles_rot['human'] = df['human']
df_angles_rot['rotation'] = rotation
df_angles_rot['rotated'] = df_rot_index # df_rot_index = 0 (if no rotation happened i.e. if we have no torso, 1 otherwise)

df_angles.to_csv(relative_path + 'Data_csv/' + dataset + '_angles.txt', sep='\t')
df_angles_rot.to_csv(relative_path + 'Data_csv/' + dataset + '_angles_rotated.txt', sep='\t')

df_angles_rot_with_names = df_angles_rot
df_angles_rot_with_names.columns.values[list(range(0, len(df_angles_rot_with_names.columns)-4))] = names_edges

df_angles_rot_with_names.to_csv(relative_path + 'Data_csv/' + dataset + '_angles_rotated_with_names.txt', sep='\t')

print(df_angles_rot_with_names)

print(math.cos(df_angles_rot_with_names.iloc[1,1]))
print(math.sin(df_angles_rot_with_names.iloc[1,1]))

d = df_angles_rot_with_names[df_angles_rot_with_names.columns[range(0, len(df_angles_rot_with_names.columns)-4)]]

def sin(x):
  return np.sin(x)

df_angles_rot_sin = d.apply(sin)
df_angles_rot_sin.columns = [i + '_sin' for i in df_angles_rot_sin.columns]
print(df_angles_rot_sin)

def cos(x):
  return np.cos(x)

df_angles_rot_cos = d.apply(cos)
df_angles_rot_cos.columns = [i + '_cos' for i in df_angles_rot_cos.columns]
print(df_angles_rot_cos)

df_angles_rot_sin_cos = pd.concat([df_angles_rot_sin, df_angles_rot_cos], axis=1)
print(df_angles_rot_sin_cos)

df_angles_rot_sin_cos['image'] = df['image']
df_angles_rot_sin_cos['human'] = df['human']
df_angles_rot_sin_cos['rotation_sin'] = np.sin(rotation)
df_angles_rot_sin_cos['rotation_cos'] = np.cos(rotation)
df_angles_rot_sin_cos['rotated'] = df_rot_index # df_rot_index = 0 (if no rotation happened i.e. if we have no torso, 1 otherwise)

df_angles_rot_sin_cos.to_csv(relative_path + 'Data_csv/' + dataset + '_angles_sin_cos.txt', sep='\t')
