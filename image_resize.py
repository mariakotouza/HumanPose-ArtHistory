import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import PIL

relative_path = ''
Image.MAX_IMAGE_PIXELS = None
image = plt.imread(relative_path + 'Data/Caracaggio2/img/Death_of_the_Virgin-Caravaggio_(1606).jpg')
print(image.shape)

baseheight = 1560
img = Image.open(relative_path + 'Data/Caracaggio2/img/Death_of_the_Virgin-Caravaggio_(1606).jpg')
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
img.save(relative_path + 'Data/Caracaggio2/img/Death_of_the_Virgin-Caravaggio_(1606)2.jpg')
