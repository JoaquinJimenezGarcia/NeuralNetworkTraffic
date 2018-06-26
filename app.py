import tensorflow as tf
import os
import skimage.data as imd
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage import transform
from skimage.color import rgb2gray

def load_ml_data(data_directory):
    dirs = [d for d in os.listdir(data_directory)
           if os.path.isdir(os.path.join(data_directory, d))]
    
    labels = []
    images = []
    
    for d in dirs:
        label_dir = os.path.join(data_directory, d)
        file_names = [os.path.join(label_dir, f)
                     for f in os.listdir(label_dir)
                     if f.endswith(".ppm")]
        
        for f in file_names:
            images.append(imd.imread(f))
            labels.append(int(d))
        
    return images, labels

main_dir = "./datasets/"
train_data_dir = os.path.join(main_dir, "Training")
test_data_dir = os.path.join(main_dir, "Testing")

images, labels = load_ml_data(train_data_dir)

images = np.array(images)
labels = np.array(labels)

w = 9999
h = 9999

for image in images:
    if image.shape[0] < h:
        h = image.shape[0]
        
    if image.shape[1] < w:
        w = image.shape[1]

images30 = [transform.resize(image, (30,30)) for image in images]

images30 = np.array(images30)
images30 = rgb2gray(images30)

x = tf.placeholder(dtype = tf.float32, shape = [None, 30, 30])
y = tf.placeholder(dtype = tf.int32, shape = [None])

images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

train_opt = tf.train.AdamOptimizer(learning_rate = 0.001). minimize(loss)

final_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(final_pred, tf.float32))

tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(600):
    _, accuracy_val = sess.run([train_opt, accuracy], feed_dict = {
        x: images30,
        y: list(labels)})
    
    if i%100 == 0:
        print("EPOCH ", i)
        print("Eficacia: ", accuracy_val)

sample_idx = random.sample(range(len(images30)), 1)
sample_images = [images30[i] for i in sample_idx]
sample_labels = [labels[i] for i in sample_idx]

sample_images = [images30[30] for i in sample_idx]
sample_labels = [labels[30] for i in sample_idx]

prediction = sess.run([final_pred], feed_dict = {x:sample_images})[0]

plt.figure(figsize=(16, 10))

for i in range(len(sample_images)):
    truth = sample_labels[i]
    predic = prediction[i]
    
    plt.subplot(10, 6, i+1)
    plt.axis("off")
    color = "green" if truth == predic else "red"
    plt.text(32, 15, "Real:      {0}\nPrediccion: {1}".format(truth, predic),
            fontsize = 14, color = color)
    plt.imshow(sample_images[i], cmap = "gray")
    
plt.show()

test_images, test_labels = load_ml_data(test_data_dir)
test_images30 = [transform.resize(im, (30, 30)) for im in test_images]
test_images30 = rgb2gray(np.array(test_images30))

prediction = sess.run([final_pred], feed_dict={x:test_images30})[0]
match_count = sum([int(l0 == lp) for l0, lp in zip(test_labels, prediction)])

acc = match_count/len(test_labels)*100
print("Eficacia de la red neuronal: {:.2f} %".format(acc))
