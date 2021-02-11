#!/usr/bin/env python3

from pprint import pprint
from time import time
import logging

from mimic3preprocess.neural_readers import InHospitalMortalityReader
from in_hospital_mortality.custom_metrics import mortality_rate_at_k
from in_hospital_mortality import neural_utils
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os
import json
import warnings
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from in_hospital_mortality.neural_model.data_handler import DataHandler
from in_hospital_mortality.neural_model.models import create_grud_model, load_grud_model
from in_hospital_mortality.neural_model.nn_utils.callbacks import ModelCheckpointwithBestWeights

def read_and_extract_matrices(reader):

    #data = neural_utils.read_chunk(reader, reader.get_number_of_examples())
    data = neural_utils.read_chunk(reader, 10)
    return data

def custom_scaler(data, _mean, _std):

    data = np.array([np.divide(mat - _mean, _std) for mat in data])
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--period_length', type=float, default=24.0, help='specify the period of prediction',
                        choices=[24.0, 48.0, -1])
    '''parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'notes', 'all_but_notes'])'''
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='/data/physician_notes/new_experiments/in_hospital_mortality_24/')
    parser.add_argument('--recurrent_dim', default='64',
                        type=lambda x: x and [int(xx) for xx in x.split(',')] or [])
    parser.add_argument('--hidden_dim', default='64',
                        type=lambda x: x and [int(xx) for xx in x.split(',')] or [])
    parser.add_argument('--model', default='GRUD', 
                        choices=['GRUD', 'GRUforward', 'GRU0', 'GRUsimple'])
    parser.add_argument('--use_bidirectional_rnn', default=False)
    parser.add_argument('--pretrained_model_file', default=None,
                        help='If pre-trained model is provided, training will be skipped.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=str, help = 'Path to the directory to store all the trained models',
                        default='/data/physician_notes/neural_results/')
    args = parser.parse_args()
    args.period_length = float('inf') if args.period_length == -1 else args.period_length
    print (args)

    warnings.filterwarnings('ignore')    

    with open(os.path.join(os.path.dirname(__file__), "../resources/channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())

    if args.period_length == 24.0:
        time_period = '24'
    elif args.period_length == 48.0:
        time_period = '48'
    else:
        time_period = 'retro'

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                             channel_info=channel_info,
                                             listfile=os.path.join(args.data, 'train_listfile.csv'),
                                             period_length=args.period_length)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                           channel_info=channel_info,
                                           listfile=os.path.join(args.data, 'val_listfile.csv'),
                                           period_length=args.period_length)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            channel_info=channel_info,
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=args.period_length)

    print('Reading data and extracting features ...')
    train = read_and_extract_matrices(train_reader)
    print ('Finished reading training data ...')
    val = read_and_extract_matrices(val_reader)
    print ('Finished reading validation data ...')
    test = read_and_extract_matrices(test_reader)
    print ('Finished reading testing data ...')
    print ('  train data shape = {}'.format(train['input'].shape[0]))
    print ('  validation data shape = {}'.format(val['input'].shape[0]))
    print ('  test data shape = {}'.format(test['input'].shape[0]))

    print('Normalizing the data to have zero mean and unit variance ...')
    mean_ = np.nanmean(np.concatenate(train['input']), axis = 0)
    std_ = np.nanstd(np.concatenate(train['input']), axis = 0)
    std_[std_ == 0] = 1
    train['input'] = custom_scaler(train['input'], mean_, std_)
    val['input'] = custom_scaler(val['input'], mean_, std_)
    test['input'] = custom_scaler(test['input'], mean_, std_)

    dataset = DataHandler(train, val, test)
    
    if K.backend() == 'tensorflow':
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = .1
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))

    if args.pretrained_model_file is not None:
        model = load_grud_model(args.pretrained_model_file)
    else:
        model = create_grud_model(input_dim = dataset.input_dim,
                                  output_dim = dataset.output_dim,
                                  output_activation = dataset.output_activation,
                                  recurrent_dim = args.recurrent_dim,
                                  hidden_dim = args.hidden_dim,
                                  predefined_model = args.model,
                                  use_bidirectional_rnn = args.use_bidirectional_rnn)

        print (model.summary)
        model.compile(optimizer='adam', loss=dataset.loss_function)
        model.fit_generator(generator=dataset.training_generator(batch_size=args.batch_size),
                            steps_per_epoch=dataset.training_steps(batch_size=args.batch_size),
                            epochs=args.epochs,
                            verbose=1,
                            validation_data=dataset.validation_generator(batch_size=args.batch_size),
                            validation_steps=dataset.validation_steps(batch_size=args.batch_size),
                            callbacks=[
                                EarlyStopping(patience=args.early_stopping_patience),
                                ModelCheckpointwithBestWeights(
                                    file_dir=os.path.join(args.save_model, 'model' + '_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
                                ),
                                TensorBoard(
                                    log_dir=os.path.join(args.save_model, 'tensorboard/', 'tb_logs' + '_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))), 
                            ]
                            )
        model.save(os.path.join(args.save_model, 'model_{}.h5'.format(time_period)))

    true_y_list = [
        dataset.training_y(), dataset.validation_y(), dataset.testing_y()
    ]

    pred_y_list = [
        model.predict_generator(dataset.training_generator_x(batch_size=args.batch_size),
                                steps=dataset.training_steps(batch_size=args.batch_size)),
        model.predict_generator(dataset.validation_generator_x(batch_size=args.batch_size),
                                steps=dataset.validation_steps(batch_size=args.batch_size)),
        model.predict_generator(dataset.testing_generator_x(batch_size=args.batch_size),
                                steps=dataset.testing_steps(batch_size=args.batch_size)),
    ]

    auc_scores = [roc_auc_score(ty, py) for ty, py in zip(true_y_list, pred_y_list)]

    print ("ROC AUC Score on Train Set:")
    print (auc_scores[0])
    print ("ROC AUC Score on Validation Set:")
    print (auc_scores[1])
    print ("ROC AUC Score on Test Set:")
    print (auc_scores[2])

    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(true_y_list[2], pred_y_list[2], K))
