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

    def __init__(self, dataset_dir, listfile=None, period_length = 24.0):
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

    def _read_timeseries(self, ts_filename):
        note_ranges = ['900001','900002','900003','900004','900005','900006','900007','900008','900010','900011','900012','900013','900014','900015']
        ret = []
        file_name = os.path.join(self._dataset_dir, ts_filename)
        tsfile_df = pd.read_csv(file_name, low_memory = False)
        tsfile_df = tsfile_df.replace(np.nan, '', regex = True)
        hours = tsfile_df.pop('Hours')
        tsfile_structured = tsfile_df.drop(columns = note_ranges)
        tsfile_note = tsfile_df[note_ranges]
        tsfile_structured = tsfile_structured.reindex(sorted(tsfile_structured.columns, key=lambda x: int(x)), axis=1)
        tsfile_note = tsfile_note.reindex(sorted(tsfile_note.columns, key=lambda x: int(x)), axis = 1)
        tsfile_structured.insert(0, 'Hours', hours)
        tsfile_note.insert(0,'Hours', hours)
        return (tsfile_structured.values, tsfile_note.values, tsfile_structured.columns, tsfile_note.columns)

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
        (X_structured, X_note, header_structured, header_note) = self._read_timeseries(name)

        return {"X": X_structured,"t": t,"y": y,"header": header_structured,"name": name}, {"X": X_note,"t": t,"y": y,"header": header_note,"name": name}
