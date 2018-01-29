# encoding: utf-8

"""
@author: huguyuehuhu
@time: 17-9-23 下午11:56
"""
from data_select import *
from data_provider.utils_mnist import load_mnist
from tensorflow.python.framework import dtypes


def main():
    action_params = {'train': True,
                     'load': False,
                     'valid': True,
                     'test': True}
    train_param = {'data_set': 'MNIST',
                   'model_type': 'MLP',
                   'should_save_logs': True,
                   'should_save_model': True,
                   'renew_logs': True}
    super_params = {'max_epoch': 100,
                    'learning_rate': 0.01,
                    'batch_size': 100,
                    }

    # Load the dataset
    print("Loading data...")
    from tensorflow.examples.tutorials.mnist import input_data
    #mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    mnist = load_mnist(train_dir='/data/gyhu/code/mnist/', dtype=dtypes.float32,
                       one_hot=True, reshape=True, validation_size=5000)
    data = mnist
    # train_data, val_data, test_data = data_provider.load_dataset(debug=False, data_name='MNIST')

    # create models & count trainable parameters
    model = MultiPerceptron(train_param=train_param)
    total_parameters = model.count_trainable_params
    print("Total training params: %.1fM" % (total_parameters / 1e6))

    if action_params['load']:
        model.load_model()
    elif action_params['train']:
        train_all_epochs(model, data=data, super_params=super_params, action_params=action_params )

    if action_params['test']:
        print("Testing...")
        mean_loss, mean_accuracy = test_one_epoch(
            model, data.test, super_params['batch_size'])
        print("mean cross_entropy: %f, mean accuracy: %f" % (
            mean_loss, mean_accuracy))

if __name__ == '__main__':
    main()