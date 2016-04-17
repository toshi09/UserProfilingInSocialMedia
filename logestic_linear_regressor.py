#!/usr/bin/python
import argparse
import numpy
import os
from skimage import io
from skimage.color import rgb2gray
from skimage import transform

import tensorflow as tf
from xml.etree import ElementTree

def get_age_group(age):
    if 18 <= age <= 24.0  :
        return "xx-24"
    elif 25 <= age <= 34:
        return "25-34"
    elif 35 <= age <= 49:
        return "35-49"
    elif 50 <= age:
        return "50-xx"
    else:
        return "xx-24"
        
def write_out(out_dir, predictions):
    """
    """
    for result in predictions:
        id, gender, age = result
        out_file = os.path.join(out_dir, id+".xml")
        
        with open(out_file, "w") as fd:
            attrs = { "userId" : id,
                      "gender" : "female" if gender else "male",
                      "age_group" : get_age_group(age),
		      'extrovert': str(3.4869),
                      'neurotic': str(2.7324),
                      'agreeable': str(3.5839),
                      'conscientious': str(3.4456),
                      'open': str(3.9087)
                    }
            tree = ElementTree.Element("user", attrs)
            fd.write(ElementTree.tostring(tree))


def getimagevec(basedir):
    """
    Generates :Training data
    tuple of profileId, sex, numpy array
    """
    profile_f = os.path.join(basedir, "training/profile", "profile.csv")
    
    with open(profile_f) as fd:
        fd.readline()
        for line in fd:
            values = line.strip().split(",")
            age = float(values[2])
            sex = float(values[3])
            userid = values[1]
            image_file = os.path.join(basedir, "training", "image", userid+".jpg")
            numpy_array = io.imread(image_file, as_grey=True)
            #print(numpy_array.shape)
            #convert_to_gray(numpy_array)
            yield(userid, sex, numpy_array, age)

def testdata(basedir):
    profile_f = os.path.join(basedir, "profile", "profile.csv")
    
    with open(profile_f) as fd:
        fd.readline()
        for line in fd:
            values = line.strip().split(",")
            userid = values[1]
            image_file = os.path.join(basedir,  "image", userid+".jpg")
            numpy_array = io.imread(image_file, as_grey=True)
            
            yield(userid, numpy_array)

gg  = getimagevec("..")
tt = testdata("..")

def flatten(vec):
    """
    """
    zz = vec.flatten()
    size = 50 * 50
    zero_v = numpy.zeros(50 * 50)
    
    for idx in range(min(len(zz), size)):
        zero_v[idx] =  zz[idx]
        
    return zero_v

def predict(in_arr, WW, bb):
    """
    """
    input = flatten(in_arr)
    out = tf.matmul(WW, in_arr) + bb
    return out

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

def train_neural_net(data_gen, test_gen):
    """
    """
    xx = tf.placeholder("float", shape=[None, 28 * 28])
    yy = tf.placeholder("float", shape=[None, 2])
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(xx, [-1, 28, 28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 2])

    b_fc2 = bias_variable([2])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    cross_entropy = -tf.reduce_sum(yy * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(yy,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    train_y = []
    train_x = []
    for idx, data in enumerate(data_gen):
        #if idx > 100:
        #    break
        _, sex, n_arr, age = data
        sex = float(sex)
        train_x.append(flatten(n_arr))
        train_y.append(numpy.array([1.0, 0.0])  if sex else numpy.array([0.0, 1.0]))

    print("Starting the training")
    print(len(train_x))
    print(len(train_y))
    
    sess.run(train_step, feed_dict={xx: train_x, yy : train_y, keep_prob: 0.5})
    print("finished training")
    
    print sess.run(accuracy, feed_dict={xx:train_x, yy: train_y, keep_prob: 1.0})

    test_x = []
    for _, numpy_arr in test_gen:
        test_x.append(flatten(numpy_arr))

    yy_p = sess.run(y_conv, feed_dict={xx: test_x, keep_prob: 1.0})
    gender = []
    for item in yy_p:
        print(yy_p)
        gender.append(1.0 if yy_p[0] > yy_p[1] else 0.0)


    return gender

    
    
def train_and_test(data_gen, test_gen):
    """
    """
    dim = 50 * 50
    xx = tf.placeholder(tf.float32, [None, dim])
    WW = tf.Variable(tf.zeros([dim, 2]))
    bb = tf.Variable(tf.zeros([2]))
    yy = tf.nn.softmax(tf.matmul(xx, WW) + bb)

    yy_a = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = -tf.reduce_sum(yy_a *  tf.log(yy))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    train_y = []
    train_x = []
    for idx, data in enumerate(data_gen):
        #if idx > 300:
        #    break
        _, sex, n_arr, age = data
        sex = float(sex)
        train_x.append(flatten(n_arr))
        train_y.append(numpy.array([1.0, 0.0])  if sex else numpy.array([0.0, 1.0]))

    print("Init Training")
    #print(train_x)
    sess.run(train_step, feed_dict={xx: train_x, yy_a: train_y})
    wt = sess.run(WW)
    bias = sess.run(bb)
    #pred = sess.run(yy)
    #print(wt)
    
    correct_prediction = tf.equal(tf.argmax(yy_a,1), tf.argmax(yy,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print sess.run(accuracy, feed_dict={xx: train_x, yy_a: train_y})

    test_uid = []
    test_x = []
    for _, numpy_arr in test_gen:
        test_x.append(flatten(numpy_arr))

    yy_p = sess.run(yy, feed_dict={xx: test_x})
    gender = []
    for item in  yy_p:
       gender.append(0.0 if item[0] > item[1] else 1.0)
    return gender

def train_age(data_gen, test_gen):
    """
    """
    dim = 50 * 50
    xx = tf.placeholder(tf.float32, [None, dim])
    WW = tf.Variable(tf.zeros([dim, 1]))
    bb = tf.Variable(tf.zeros([1]))
    yy = tf.add(tf.matmul(xx, WW), bb)

    yy_a = tf.placeholder(tf.float32, [None, 1])
 
    cost = tf.pow(yy_a - yy, 2)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    train_y = []
    train_x = []
    for idx, data in enumerate(data_gen):
        if idx > 100:
            break
        _, sex, n_arr, age = data
        age = float(age)
        train_x.append(flatten(n_arr))
        train_y.append(numpy.array([age]))

    print("Init Training")
 
    sess.run(train_step, feed_dict={xx: train_x, yy_a: train_y})
     
    test_x = []
    for _, numpy_arr in test_gen:
        test_x.append(flatten(numpy_arr))

    yy_p = sess.run(yy, feed_dict={xx: test_x})
    age = []
    for item in  yy_p:
        age.append(item[0]/ 100000.0)
    return age

def parse_args():
    parser = argparse.ArgumentParser(description="""Script takes full input path to
                         test directory, output directory and training directory""")

    parser.add_argument('-d',
                        "--training_dir",
                        default='',
                        type=str, 
                        help='Full path to input trainig directory')

    parser.add_argument('-i',
                        "--test_dir",
                        type=str, 
                        required=True,
                        help='Full path to input test directory containing profile and text dir')
                        
    parser.add_argument('-o', "--output_dir",
                        type=str,
                        default='output',
                        help='The path to output directory')
                    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test_dir = args.test_dir
    uids = []
    for id, _ in testdata(test_dir):
        uids.append(id)
        
 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    gender = train_and_test(getimagevec(args.training_dir), testdata(test_dir))
    age = train_age(getimagevec(args.training_dir), testdata(test_dir))
    write_out(args.output_dir, zip(uids, gender, age))
