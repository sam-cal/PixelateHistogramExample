#!/usr/bin/env python
import tensorflow as tf
from PixelateHistogram.layers import *

print("Very simple histogram passing trough 1-layer PixelateLayer")
PL1 = PixelateLayer(stepSize=0.1, flatten=False)
	
histo_1 = tf.constant([[[1000], [0], [0]]])
pixels =PL1(histo_1)
print (histo_1, '->', pixels)


print("Very simple histogram passing trough 3-layer PixelateLayer")
PL2 = PixelateLayer(stepSize=0.1, n_sigma=1, flatten=False)

histo_2 = tf.constant([[[1000], [100], [100]]])
pixels =PL2(histo_2)
print (histo_2, '->', pixels)

print("... and same PDF with 1000 times more stat:")
PL3 = PixelateLayer(stepSize=0.1, n_sigma=1, flatten=False)

histo_3 = tf.constant([[[1e6], [1e5], [1e5]]])
pixels =PL3(histo_3)
print (histo_3, '->', pixels)


print("Same histogram with default settings::")
PL4 = PixelateLayer()

pixels =PL4(histo_3)
print (histo_3, '->', pixels)
