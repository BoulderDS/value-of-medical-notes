#!/usr/bin/env python3

import os
import numpy as np

__all__ = ['DataHandler']


class DataHandler(object):

    def __init__(self, train, val, test):
    
        super(DataHandler, self).__init__()
        self._input_dim = None
        self._output_dim = None
        self._output_activation = None
        self._loss_function = None

        self._load_data(train, val, test)

    def _load_data(self, train, val, test):

        self._data = {}
        self._data['train'] = train
        self._data['val'] = val
        self._data['test'] = test
        
        self._input_dim = self._data['train']['input'][0].shape[-1]
        self._output_dim = 1
        self._output_activation = 'sigmoid'
        self._loss_function = 'binary_crossentropy'

    def _get_generator(self, split_type, shuffle, batch_size, return_targets):

        if not return_targets and shuffle:
            raise ValueError('Do not shuffle when targets are not returned.')
        if batch_size != 1:
            raise ValueError('This program is not yet compatible for batch sizes greater than 1')
        data = {}
        data['input'] = np.copy(self._data[split_type]['input'])
        data['masking'] = np.copy(self._data[split_type]['masking'])
        data['timestamp'] = np.copy(self._data[split_type]['timestamp'])
        data['label'] = np.copy(self._data[split_type]['label'])

        split_len = data['input'].shape[0]

        def _generator():
            while True:
                if shuffle:
                    indexes = np.random.permutation(split_len)
                else:
                    indexes = np.arange(split_len)

                batch_id = 0
                while batch_id < split_len:
                    inputs = [np.array([data[s][indexes[batch_id]]]) for s in ['input','masking','timestamp']]
                    print (inputs)
                    targets = data['label'][indexes[batch_id]]
                    yield (inputs, targets)
                    batch_id += batch_size
                    print ('.', end='')

        def _inputs_generator():
            for inputs, _ in _generator():
                yield inputs

        if not return_targets:
            return _inputs_generator()
        return _generator()

    def training_generator(self, batch_size):
        return self._get_generator(split_type='train',shuffle=True,
                                   batch_size=batch_size, return_targets=True)

    def validation_generator(self, batch_size):
        return self._get_generator(split_type='val',shuffle=False,
                                   batch_size=batch_size, return_targets=True)

    def testing_generator(self, batch_size):
        return self._get_generator(split_type='test',shuffle=False,
                                   batch_size=batch_size, return_targets=True)

    def _steps(self, split_type, batch_size):
        return int(self._data[split_type]['input'].shape[0] / batch_size)

    def training_steps(self, batch_size):
        return self._steps(split_type='train', batch_size = batch_size)

    def validation_steps(self, batch_size):
        return self._steps(split_type='val', batch_size = batch_size)

    def testing_steps(self, batch_size):
        return self._steps(split_type='test', batch_size = batch_size)

    def training_y(self):
        return self._data['train']['label']

    def validation_y(self):
        return self._data['val']['label']

    def testing_y(self):
        return self._data['test']['label']

    def training_generator_x(self, batch_size):
        return self._get_generator(split_type='train',shuffle=False,
                                   batch_size=batch_size, return_targets=False)

    def validation_generator_x(self, batch_size):
        return self._get_generator(split_type='val',shuffle=False,
                                   batch_size=batch_size, return_targets=False)

    def testing_generator_x(self, batch_size):
        return self._get_generator(split_type='test',shuffle=False,
                                   batch_size=batch_size, return_targets=False)   

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def output_activation(self):
        return self._output_activation

    @property
    def loss_function(self):
        return self._loss_function
