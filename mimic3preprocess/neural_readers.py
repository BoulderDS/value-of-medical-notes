#!/usr/bin/env python3

import os
import numpy as np
import random
import pandas as pd

class Reader(object):

    def __init__(self, dataset_dir, listfile = None):

        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):

        return len(self._data)

    def random_shuffle(self, seed=None):

        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)

class InHospitalMortalityReader(Reader):

    def __init__(self, dataset_dir, channel_info, listfile=None, period_length = 24.0):
        """ Reader for in-hospital moratality prediction task.
        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length
        self._channel_info = channel_info

    def convert_text_to_numeric(self, data, header):
        
        '''
        convert data from structed tables that is categorical (text form) in nature to integers using a map
        '''
        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[1]):
                channel = header[j]
                if (data[i][j] != ''):
                    try:
                        if len(self._channel_info[channel]['possible_values']) != 0:
                            data[i][j] = self._channel_info[channel]['values'][data[i][j]]
                    except KeyError:
                        data[i][j] = 0
                    try:
                        data[i][j] = float(data[i][j])
                    except ValueError:
                        data[i][j] = float(0)
        return data

    '''def get_time_deltas(self, tmask, tsd, timestamps):
        for i in range(1, len(tsd)):
            if tmask[i-1] == 0:
                tsd[i] = timestamps[i] - timestamps[i-1] + tsd[i-1]
            elif tmask[i-1] == 1:
                tsd[i] = timestamps[i] - timestamps[i-1]
        return tsd'''

    def _read_timeseries(self, ts_filename):
        
        note_ranges = ['900001','900002','900003','900004','900005','900006','900007','900008','900010','900011','900012','900013','900014','900015']
        file_name = os.path.join(self._dataset_dir, ts_filename)
        tsfile_df = pd.read_csv(file_name, low_memory = False)
        tsfile_df = tsfile_df.replace(np.nan, '', regex = True)
        zeroval = tsfile_df.iloc[0]['Hours']
        tsfile_df.iloc[0, tsfile_df.columns.get_loc('Hours')] = 0
        tsfile_df['Hours'][1:] = tsfile_df['Hours'][1:] - zeroval
        tsfile_df = tsfile_df[tsfile_df['Hours'] <= 24]
        timestamps = tsfile_df.pop('Hours').values

        tsfile_structured = tsfile_df.drop(columns = note_ranges)
        tsfile_note = tsfile_df[note_ranges]
        tsfile_structured = tsfile_structured.reindex(sorted(tsfile_structured.columns, key=lambda x: int(x)), axis=1)
        tsfile_note = tsfile_note.reindex(sorted(tsfile_note.columns, key=lambda x: int(x)), axis = 1)

        tsfile_structured_values = tsfile_structured.values
        tsfile_structured_columns = tsfile_structured.columns
        tsfile_structured_values = self.convert_text_to_numeric(tsfile_structured_values, tsfile_structured_columns)
        tsfile_structured_values = pd.DataFrame(tsfile_structured_values).replace('', np.nan).values
        tsfile_structured_masks = pd.notna(tsfile_structured_values).astype(int)
        #tsfile_structured_timedeltas = np.zeros(tsfile_structured_masks.shape)
        #for i in range(tsfile_structured_timedeltas.shape[1]):
        #    tsfile_structured_timedeltas[:,i] = self.get_time_deltas(tsfile_structured_masks[:,i], tsfile_structured_timedeltas[:,i], timestamps)

        #return (timestamps, tsfile_structured_values, tsfile_structured_columns, tsfile_structured_masks, tsfile_structured_timedeltas)
        return (timestamps, tsfile_structured_values, tsfile_structured_columns, tsfile_structured_masks)

    def read_example(self, index):
        """ Reads the example with given index.
        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]        
        #(timeseries, X_structured_values, X_structured_header, X_structured_masks, X_structured_timedeltas) = self._read_timeseries(name)
        (timeseries, X_structured_values, X_structured_header, X_structured_masks) = self._read_timeseries(name)
        timeseries_2d = np.reshape(timeseries, (timeseries.shape[0], 1))

        #return {"X_t": X_structured_values,"X_t_mask": X_structured_masks, "X_t_timedeltas": X_structured_timedeltas, "timeseries": timeseries,  "t": t,"y": y,"header": X_structured_header,"name": name}
        #return {"X_t": X_structured_values,"X_t_mask": X_structured_masks, "timeseries": timeseries, "t": t,"y": y,"header": X_structured_header,"name": name}
        return {"input": X_structured_values, "masking": X_structured_masks, "timestamp": timeseries_2d, "label": y}
