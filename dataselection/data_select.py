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

    total_start_time = time.time()
    for epoch in range(max_epoch):
        print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
        start_time = time.time()

        print("training...")
        loss, accuracy = train_one_epoch(
            model, data.train, batch_size, learning_rate)
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