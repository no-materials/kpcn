import tensorflow as tf
import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

#PLY IO
