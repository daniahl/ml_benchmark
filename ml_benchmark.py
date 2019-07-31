#!/usr/bin/env python

import sys
assert sys.version_info >= (3, 4), "Python >3.4 required."

from importlib import util
assert util.find_spec('numpy') != None, "NumPy not installed."
assert util.find_spec('tqdm') != None, "tqdm not installed."

# ----------------- NumPy -----------------

import numpy as np
from time import time
from tqdm import tqdm

np.random.seed(42)

print(f'\nNumPy version: {np.__version__}\n')
np.__config__.show()

SIZE = 4096
N = 5

print('\nGenerating matrices...')
t = time()
A, B = np.random.random((SIZE, SIZE)), np.random.random((SIZE, SIZE))
C = np.random.random((int(SIZE/2), int(SIZE/4)))
print(f'It took {(time()-t):.2f} s.')

print('\nRunning matrix multiplication test...')
t = time()
for i in tqdm(range(N)):
    np.dot(A, B)
print(f'{SIZE}x{SIZE} matrix multiplication in {(time()-t)/N:.2f} s.\n')

print('Running SVD test...')
t = time()
for i in tqdm(range(N)):
    np.linalg.svd(C, full_matrices=False)
print(f'SVD of {int(SIZE/2)}x{int(SIZE/4)} matrix in {(time()-t)/N:.2f} s.\n')

if util.find_spec('sklearn') == None:
    print("Scikit-learn not installed. Skipping tests.")
    sys.exit(0)

# ----------------- scikit-learn -----------------

import sklearn
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

print(f'Scikit-learn version: {sklearn.__version__}')

N_SAMPLES = 10000
N_FEATURES = 50
N_CLASSES = 10

print(f'\nCreating classification dataset ({N_CLASSES} classes, {N_SAMPLES} samples, {N_FEATURES} features)...')
t = time()
X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=6, n_redundant=2, n_repeated=0, n_classes=N_CLASSES, n_clusters_per_class=3, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=42)
print(f'It took {(time()-t):.2f} s.\n')

clf = SVC(gamma='auto', random_state=42)
print(f'Running SVM test...')
t = time()
for i in tqdm(range(N)):
    clf.fit(X, y)
print(f'SVM trained in {(time()-t)/N:.2f} s.')
print(f'Score on training set: {clf.score(X, y)}\n')

N_ESTIMATORS = 100
MAX_DEPTH = 50

clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, n_jobs=1, random_state=42)
print('Running random forest test (1 thread)...')
t = time()
for i in tqdm(range(N)):
    clf.fit(X, y)
print(f'Random forest trained in {(time()-t)/N:.2f} s.')
print(f'Score on training set: {clf.score(X, y)}\n')

clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, n_jobs=2, random_state=42)
print('Running random forest test (2 threads)...')
t = time()
for i in tqdm(range(N)):
    clf.fit(X, y)
print(f'Random forest trained in {(time()-t)/N:.2f} s.')
print(f'Score on training set: {clf.score(X, y)}\n')

clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, n_jobs=-1, random_state=42)
print('Running random forest test (max threads)...')
t = time()
for i in tqdm(range(N)):
    clf.fit(X, y)
print(f'Random forest trained in {(time()-t)/N:.2f} s.')
print(f'Score on training set: {clf.score(X, y)}\n')

if util.find_spec('tensorflow') == None:
    print("Tensorflow not installed. Skipping tests.")
    sys.exit(0)

# ----------------- TensorFlow -----------------

import tensorflow as tf
print(f'Tensorflow version: {tf.__version__}')
if not tf.test.gpu_device_name():
    print('No GPU found. Using CPU.')
tf.logging.set_verbosity(tf.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'                # Uncomment to force CPU use on GPU-enabled systems
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, applications
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

tf.random.set_random_seed(42)

IMAGES = 20

print('\nGenerating image data...')
t = time()
images = np.random.random((IMAGES, 224, 224, 3))
labels = np.random.randint(0, 9, (IMAGES, 1))
print(f'It took {(time()-t):.2f} s.\n')

input_tensor = Input(shape=(224, 224, 3))
base_model = applications.ResNet50(include_top=False, input_tensor=input_tensor)
x = GlobalAveragePooling2D()(base_model.output)
final_output = Dense(10, activation='softmax')(x)
model = Model(input_tensor, final_output)

model.compile(optimizer=optimizers.Adam(lr=1e-3), loss="sparse_categorical_crossentropy",  metrics=['accuracy'])
print(f'Training 5 epochs of 10-class ResNet50 on ({IMAGES}, 224, 224, 3)...')
t = time()
model.fit(images, labels, batch_size=2, epochs=5)
print(f'Training took {(time()-t):.2f} s.\n')
