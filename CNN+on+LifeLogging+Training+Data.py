
# coding: utf-8

# In[1]:

########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
from random import shuffle
import numpy as np
from scipy.misc import imread, imresize
import time
start_time = time.time()

with tf.device('/gpu:1'):
    class vgg16:
        def __init__(self, imgs, weights=None, sess=None):
            self.imgs = imgs
            self.convlayers()
            self.fc_layers()
            self.probs = tf.nn.softmax(self.fc3l)
            
            #self.raw_output is self.fc3l w/o a softmax applied on it.  
            #This is because a softmax is accounted for later
            self.raw_output = self.fc3l
            
            if weights is not None and sess is not None:
                self.load_weights(weights, sess)


        def convlayers(self):
            self.parameters = []

            # zero-mean input
            with tf.name_scope('preprocess') as scope:
                mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                images = self.imgs-mean

            # conv1_1
            with tf.name_scope('conv1_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv1_2
            with tf.name_scope('conv1_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool1
            self.pool1 = tf.nn.max_pool(self.conv1_2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool1')

            # conv2_1
            with tf.name_scope('conv2_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv2_2
            with tf.name_scope('conv2_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool2
            self.pool2 = tf.nn.max_pool(self.conv2_2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool2')

            # conv3_1
            with tf.name_scope('conv3_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_2
            with tf.name_scope('conv3_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_3
            with tf.name_scope('conv3_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool3
            self.pool3 = tf.nn.max_pool(self.conv3_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool3')

            # conv4_1
            with tf.name_scope('conv4_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_2
            with tf.name_scope('conv4_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_3
            with tf.name_scope('conv4_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool4
            self.pool4 = tf.nn.max_pool(self.conv4_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool4')

            # conv5_1
            with tf.name_scope('conv5_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_2
            with tf.name_scope('conv5_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_3
            with tf.name_scope('conv5_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool5
            self.pool5 = tf.nn.max_pool(self.conv5_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool4')

        def fc_layers(self):
            # fc1
            with tf.name_scope('fc1') as scope:
                shape = int(np.prod(self.pool5.get_shape()[1:]))
                fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                             dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
                fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                     trainable=True, name='biases')
                pool5_flat = tf.reshape(self.pool5, [-1, shape])
                fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
                self.fc1 = tf.nn.relu(fc1l)
                self.parameters += [fc1w, fc1b]

            # fc2
            with tf.name_scope('fc2') as scope:
                fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                             dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
                fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                     trainable=True, name='biases')
                fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
                self.fc2 = tf.nn.relu(fc2l)
                self.parameters += [fc2w, fc2b]

            # fc3
            with tf.name_scope('fc3') as scope:
                
                #we change the shape to [4096, 2] because there should be 2 classes in my case
                #If we want more than two classes, we can always change this number
                fc3w = tf.Variable(tf.truncated_normal([4096, 2],
                                                             dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
                
                #Shape here is also changed because of the reason stated above
                fc3b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),
                                     trainable=True, name='biases')
                self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
                self.parameters += [fc3w, fc3b]

        def load_weights(self, weight_file, sess):
            weights = np.load(weight_file)
            keys = sorted(weights.keys())
            for i, k in enumerate(keys):
                #op=tf.initialize_variables([self.parameters[i]])
                op = tf.variables_initializer([self.parameters[i]])

                sess.run(op)
                param=sess.run(self.parameters[i])
                
                #If the shapes are equal, use preloaded weights, otherwise, use random weights
                if param.shape == np.shape(weights[k]):
                    sess.run(self.parameters[i].assign(weights[k]))
                    print(i, k, np.shape(weights[k]), 'using preloaded weights ')
                else:
                    print(i,k,np.shape(weights[k]), 'using random weights!')



# In[ ]:

if __name__ == '__main__':
   
    
   
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #saver = tf.train.import_meta_graph('/home/data2/vision6/azlu/my_models/my_model_125.ckpt.meta')
        #saver.restore(sess, "/home/data2/vision6/azlu/my_models/my_model_125.ckpt")
        
        #two placeholders, we will use these later
        #lbls has a shape of two because the label will either be 0 or 1
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        lbls = tf.placeholder(tf.float32, [None, 2])
        
        
        #vgg is now the vgg16 preloaded weights
        vgg = vgg16(imgs, '/home/data2/vision6/azlu/vgg16_weights.npz', sess)

        #We use softmax and cross entropy here.  Notice that is is applied on self.raw_output
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lbls, logits=vgg.raw_output))

        #We use GD Optimizer.  The learning rate has to be this small to reduce errors in the final output
        '''
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.0005
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 191, 0.33, staircase=True)
        '''
        learning_rate=0.0005
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)  
        #We could also use Adam Optimizer, this code is solved yet but it'll look a bit like:
        #train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cross_entropy)

        #We want the prediction (vgg.probs) to equal the correct label (lbls), with a one-hot vector encoding
        correct_prediction = tf.equal(tf.argmax(vgg.probs, 1), tf.argmax(lbls, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #these two will be used to calculate the overall accuracy later
        sum_of_acc = 0
        num_of_acc = 0

        sum_of_loss = 0
        num_of_loss = 0

        #1904 images, batches, of 10, so 191 total iterations to get through all the iamges
        num_batches=192

        #We use this to shuffle the batches 1 more time
        rand_batches = [i for i in range(1, num_batches)]
        shuffle(rand_batches)
        
        

        #Through testing, we found 22 epoches provides the highest accuracy
        for j in range(65):

            #-----------------------------------TRAINING-------------------------------------
            for i in rand_batches:

                print("batch ", j, i)

                #This part loads in the image data in batches, wherever they may be
                img_batch=np.load('/home/data2/vision6/azlu/training_data_10/img_tr_batch_{0}.npz'.format(i))
                image_data = img_batch['images']

                #This part loads in the label data in corresponding batches, wherever they may be
                lbl_batch=np.load('/home/data2/vision6/azlu/training_data_10/label_tr_batch_{0}.npz'.format(i))
                label_data = lbl_batch['labels']

                #Actual training step
                sess.run(train_step, feed_dict={imgs: image_data, lbls: label_data})
                
                
                
                #Calculate the loss
                sum_of_loss += sess.run(cross_entropy, feed_dict={imgs: image_data, lbls: label_data})
                num_of_loss = num_of_loss+1
                
                #
                
                #Everything below is stuff that could be helpful to document along the way to ensure
                #that everything is outputting correctly
            
            '''
                print('Network output probs:')
                print(sess.run(vgg.raw_output, feed_dict={imgs: image_data}))
                print(sess.run(vgg.probs, feed_dict={imgs: image_data}))
                print(sess.run(tf.argmax(vgg.probs, 1), feed_dict={imgs: image_data}))

                print('Labels:')
                print(sess.run(tf.stack(label_data)))
                print(sess.run(tf.argmax(label_data, 1)))

                print('Correct Predictions:')
                print(sess.run(correct_prediction, feed_dict={imgs: image_data, lbls: label_data}))
            '''
       
            #Calculate the loss
            #print('Loss:', sum_of_loss/num_of_loss)
            #print()
            #------------------------------------TESTING--------------------------------------------
            for i in range(1,7):   

                #Again load in the data
                img_test_batch=np.load('/home/data2/vision6/azlu/val_data/img_val_batch_{0}.npz'.format(i))
                image_test_data = img_test_batch['images']

                lbl_test_batch=np.load('/home/data2/vision6/azlu/val_data/label_val_batch_{0}.npz'.format(i))
                label_test_data = lbl_test_batch['labels']

                #The per batch accuracy along the way
                print('Batch Accuracy:', repr(sess.run(accuracy, feed_dict={imgs: image_test_data, lbls: label_test_data})))

                #Adjusting total accuracy parameters along the way
                sum_of_acc += sess.run(accuracy, feed_dict={imgs: image_test_data, lbls: label_test_data})
                num_of_acc = num_of_acc+1                

            #Total accuracy
            print('Total Accuracy:', sum_of_acc/num_of_acc)
            print("EPOCH ", j, ": --- %s seconds ---" % (time.time() - start_time))
            print()
            
            saver = tf.train.Saver()
           
            savePath=saver.save(sess, '/home/data2/vision6/azlu/my_models/take3/my_model_'+str(j)+'.ckpt')


# In[ ]:



