# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:36:13 2019

@author: Mary
"""
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras import preprocessing
from functools import partial
from datetime import datetime


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "logs/run-{}".format(now)

def vectorize(sequences, dimension = 4000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results
 
n_outputs=2

def one_hidden_layer(X, n_hidden1 = 400, n_outputs=n_outputs, activation_func=tf.nn.relu):
    with tf.name_scope("dnn"):
        hidden1=tf.layers.dense(X, n_hidden1, name="hidden_1", activation=activation_func)
        logits = tf.layers.dense(hidden1, n_outputs, name="outputs")
    return logits  

def network_with_dropout(X, n_hidden1 = 1000, n_hidden2=1000, n_hidden3=800, n_hidden4=500, n_outputs = n_outputs, activation_func=tf.nn.relu):
    training=tf.placeholder_with_default(False, shape=(), name='training')
    dropout_rate=0.5
    X_drop=tf.layers.dropout(X, dropout_rate, training=training)
    with tf.name_scope("dnn"):
        hidden1=tf.layers.dense(X_drop, n_hidden1, name="hidden_1", activation=activation_func)
        hidden1_drop=tf.layers.dropout(hidden1,dropout_rate, training=training)
        hidden2=tf.layers.dense(hidden1_drop, n_hidden2, activation=activation_func)
        hidden2_drop=tf.layers.dropout(hidden2,dropout_rate, training=training)
        hidden3=tf.layers.dense(hidden2_drop, n_hidden3, activation=activation_func)
        hidden3_drop=tf.layers.dropout(hidden3,dropout_rate, training=training)
        hidden4=tf.layers.dense(hidden3_drop, n_hidden4, activation=activation_func)
        hidden4_drop=tf.layers.dropout(hidden4,dropout_rate, training=training)
        logits = tf.layers.dense(hidden4_drop, n_outputs, name="outputs")
    return logits

scale=0.5
my_dense_layer=partial(
        tf.layers.dense, activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))

def network_with_regularization(X, n_hidden1 = 2000, n_hidden2=2000, n_hidden3=1000, n_hidden4=800, n_outputs = 2, activation_func=tf.nn.relu):
    with tf.name_scope("dnn"):
        hidden1=my_dense_layer(X, n_hidden1, name="hidden_1", activation=activation_func)
        hidden2=my_dense_layer(hidden1, n_hidden2, name="hidden_2")
        hidden3 = my_dense_layer(hidden2, n_hidden3, name="hidden_3", activation=activation_func)
        hidden4 = my_dense_layer(hidden3, n_hidden4, name="hidden_4", activation=activation_func)
        logits = my_dense_layer(hidden4, n_outputs, name="outputs")
    return logits

def multiple_hidden_layers(X, n_hidden1 = 200, n_hidden2 = 300,n_outputs = 2, activation_func=tf.nn.relu):
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden_1", activation=activation_func)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden_2", activation=activation_func)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    return logits

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def mlp_network(layers, learning_rate, epochs, batches, seed, activation_func):
    tf.reset_default_graph()
    np.random.seed(seed)
    tf.set_random_seed(seed)

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=4000)

    X_train = vectorize(X_train)    
    X_test = vectorize(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    m,n = X_train.data.shape
   
    X = tf.placeholder(tf.float32, shape=(None, n), name="X") 
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    if layers == 1:
        logits = one_hidden_layer(X=X, activation_func=activation_func)
        if layers == 1.1:
            logits = network_with_regularization(X=X, activation_func=activation_func)
        if layers== 1.2:
            logits = network_with_dropout(X=X, activation_func=activation_func)
    else:
        logits = multiple_hidden_layers(X=X, activation_func=activation_func)

    with tf.name_scope("loss"):  
        if layers == 1.1:
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)  
            base_loss = tf.reduce_mean(xentropy, name="loss")  
            loss = tf.add_n([base_loss]+reg_losses, name="loss")
        else:     
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)  
            loss = tf.reduce_mean(xentropy, name="loss")                              
            
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):           
        correct = tf.nn.in_top_k(logits, y ,1)
        accuracy = tf.reduce_mean( tf.cast(correct, tf.float32) )
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    counter=0
    
    with tf.Session() as sess:
        init.run()
        train_accuracy_summary = tf.summary.scalar("Train Accuracy", accuracy) 
        test_accuracy_summary = tf.summary.scalar("Test Accuracy", accuracy) 
        file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
        for epoch in range(epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batches):
                counter+=1
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                
                if counter%100 ==0:
                    train_summary_str = sess.run(train_accuracy_summary, feed_dict={X: X_batch, y: y_batch})
                    test_summary_str = sess.run(test_accuracy_summary, feed_dict={X: X_test, y: y_test})
                    file_writer.add_summary(train_summary_str, counter)
                    file_writer.add_summary(test_summary_str, counter)
                if counter%30==0:
                    acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch}) 
                    acc_val=accuracy.eval(feed_dict={X:X_test, y: y_test})
                    loss_train=loss.eval(feed_dict={X:X_batch, y:y_batch})  
                    loss_val=loss.eval(feed_dict={X:X_test, y: y_test})
                    print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_val, "Train loss", loss_train,"Test loss:", loss_val)
             
        save_path = saver.save(sess,"tmp/imdb-final.ckpt")
        loss_test=loss.eval(feed_dict={X: X_test, y: y_test})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        loss_train=loss.eval(feed_dict={X: X_train, y: y_train})
        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        print("Test Accuracy: {:3f}".format(acc_test))
        print("Test loss: {:3f}".format(loss_test))
        print("Train Accuracy: {:3f}".format(acc_train))
        print("Train loss: {:3f}".format(loss_train))
               


