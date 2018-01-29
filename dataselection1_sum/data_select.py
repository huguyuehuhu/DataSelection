# encoding: utf-8

"""
@author: huguyuehuhu
@time: 17-9-20 下午8:57
"""
import data_provider
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from models.model_class import MultiPerceptron


def train_one_epoch(model, data, batch_size, learning_rate):
    num_examples = data.num_examples
    total_loss = []
    total_accuracy = []
    for i in range(num_examples // batch_size):
        batch = data.next_batch(batch_size)
        images, labels = batch
        feed_dict = {
            model.images: images,
            model.labels: labels,
            model.learning_rate: learning_rate,
            model.is_training: True,
        }
        fetches = [model.updater, model.loss, model.accuracy]
        result = model.sess.run(fetches, feed_dict=feed_dict)
        _, loss, accuracy = result
        total_loss.append(loss)
        total_accuracy.append(accuracy)
    mean_loss = np.mean(total_loss)
    mean_accuracy = np.mean(total_accuracy)
    return mean_loss, mean_accuracy


def train_one_epoch_single_loss(model, data, batch_size, learning_rate, prob, epoch):
    num_examples = data.num_examples
    total_loss = []
    total_accuracy = []
    loss_table = np.zeros(data.num_examples)
    delta_w_table = np.zeros(data.num_examples)
    w_old = model.sess.run([var for var in tf.trainable_variables()])
    temp=[]
    for i in range(num_examples // batch_size):
        batch = data.next_batch_with_prob(batch_size, prob, begin_shuffle=True)
        images, labels, index = batch
        feed_dict = {
            model.images: images,
            model.labels: labels,
            model.learning_rate: learning_rate,
            model.is_training: True,
        }
        fetches = [model.updater, model.loss, model.accuracy, model.loss_single_sample]
        result = model.sess.run(fetches, feed_dict=feed_dict)
        _, loss, accuracy, loss_single = result
        total_loss.append(loss)
        total_accuracy.append(accuracy)

        loss_table[index] = loss_single
        w = model.sess.run([var for var in tf.trainable_variables()])

        # delta_w = list(map(lambda x: (x[0] - x[1]).ravel(), zip(w, w_old)))
        # d = np.array([])
        # for var in delta_w:
        #     d = np.hstack((d, var))
        # delta_w_table[index] = np.linalg.norm(d, ord=1)

        delta_w = list(map(lambda x: np.sum(np.abs(x[0] - x[1])), zip(w, w_old)))
        delta_w_table[index] = sum(np.array(delta_w))
        w_old = w
        # debug
        temp.append(sum(np.array(delta_w)))

    mean_loss = np.mean(total_loss)
    mean_accuracy = np.mean(total_accuracy)
    p_loss = loss_table/sum(loss_table)
    p_w = delta_w_table/sum(delta_w_table)
    prob_temp = np.multiply(p_loss, p_w)

    if epoch>0:
        # prob = p_loss
        # prob = p_w
        prob = prob_temp / sum(prob_temp)


    return mean_loss, mean_accuracy, prob


def test_one_epoch(model, data, batch_size):
    num_examples = data.num_examples
    total_loss = []
    total_accuracy = []
    for i in range(num_examples // batch_size):
        batch = data.next_batch(batch_size)
        images, labels = batch
        feed_dict = {
            model.images: images,
            model.labels: labels,
            model.is_training: False,
        }
        fetches = [model.loss, model.accuracy]
        loss, accuracy = model.sess.run(fetches, feed_dict=feed_dict)
        total_loss.append(loss)
        total_accuracy.append(accuracy)

        if model.should_save_logs:
            # global step
            model.batches_step += 1
            model.log_loss_accuracy(
                loss, accuracy, model.batches_step, prefix='per_batch',
                should_print=False)

    mean_loss = np.mean(total_loss)
    mean_accuracy = np.mean(total_accuracy)
    return mean_loss, mean_accuracy


def train_all_epochs(model, data, super_params, action_params):
    max_epoch = super_params['max_epoch']
    learning_rate = super_params['learning_rate']
    batch_size = super_params['batch_size']
    # initial selection prob as equal
    prob = np.ones(data.train.num_examples) * (1 / data.train.num_examples)

    total_start_time = time.time()
    for epoch in range(max_epoch):
        print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
        start_time = time.time()
        if epoch == 50:
            learning_rate = learning_rate/10
        # if epoch == 70:
        #     learning_rate = learning_rate / 10
        print("training...")
        # loss, accuracy = train_one_epoch(
        #     model, data.train, batch_size, learning_rate)

        loss, accuracy, prob = train_one_epoch_single_loss(
            model, data.train, batch_size, learning_rate, prob, epoch)

        if model.should_save_logs:
            model.log_loss_accuracy(loss, accuracy, epoch, prefix='train', should_print=True)

        if action_params['valid']:
            print("Validation...")
            loss, accuracy = test_one_epoch(
                model, data.validation, batch_size)
            if model.should_save_logs:
                model.log_loss_accuracy(loss, accuracy, epoch, prefix='valid', should_print=True)
        else:
            pass

        # time information
        time_per_epoch = time.time() - start_time
        seconds_left = int((max_epoch - epoch) * time_per_epoch)
        print("Time per epoch: %s, Est. complete in: %s" % (
            str(timedelta(seconds=time_per_epoch)),
            str(timedelta(seconds=seconds_left))))

        # save models
        if model.should_save_model:
            model.save_model()

    total_training_time = time.time() - total_start_time
    print("\nTotal training time: %s" % str(timedelta(
        seconds=total_training_time)))