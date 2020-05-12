# HumanPose-ArtHistory 
Human pose analysis in art-history: 2D pose analysis, clustering and visualization

## Project structure
The project contains three scripts:

- **preprocessing.py**: takes the .json file of a dataset exported from Supervisely and outputs a .txt file where each row contains the (x,y) coordinates for the 24 keypoints of each image, and a corresponding file with the angles of each pair of keypoints rotated according to the torso.
- **clustering.py**: clusters the datasets using k-means and creates 2D and 3D plots using PCA.
- **tsne.py**: clusters the datasets using k-means and creates 2D and 3D plots using tsne. Two cases are tested: The first one takes into account inly the keypoints of the human posture, whereas the second one uses both the keypoints ans the torso rotation angles.
- **torso_clustering.py**: clusters the datasets using only the torso rotation angles. 
- **Visualization.py**: helper functions for visualization
- **image_resize.py**: resizes an imput image

## Installation
To run the project you have to execute the following steps:

1. Create the virtual environment humanpose:
```
python -m venv humanpose
```

2. Activate the virtual environment:
#### For windows
```
humanpose\Scripts\activate.bat
```

#### For Linux
```
source humanpose/bin/activate
```

3. Install the dependencies:
```
pip install -r requirements.txt
```

4. Run the scripts
```
python preprocessing.py
python clustering.py
python tsne.py
python torso_clustering.py
```
