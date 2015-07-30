#!/usr/bin/env python
# -*- coding: utf8 -*-


"""
Dask version of
https://hdfgroup.org/wp/2015/04/putting-some-spark-into-hdf-eos/
"""

from __future__ import print_function, division

import os
import glob
import re
import csv

from itertools import chain
from time import time

import h5py

import numpy as np
import pandas as pd

import dask.bag as db

from dask import do
from dask.diagnostics import ProgressBar

from cytoolz import concat
from toolz.compatibility import map, zip

from bokeh.charts import TimeSeries, show, output_file


def data(filename):
    with h5py.File(filename, mode='r') as f:
        dset = f['/HDFEOS/GRIDS/NCEP/Data Fields/Tair_2m']
        fill = dset.attrs['_FillValue'][0]
        x = dset[:]
        cols = dset.shape[1]
    return x[x != fill], cols


rx = re.compile(r'^GSSTF_NCEP\.3\.(\d{4}\.\d{2}\.\d{2})\.he5$')


def date_from_filename(filename, rx=rx):
    return re.match(rx, filename).group(1).replace('.', '-')


def summarize(filename):
    v, _ = data(filename)
    return [(date_from_filename(os.path.basename(filename)),
             len(v), np.mean(v), np.median(v), np.std(v, ddof=1))]


def argtopk(k, x):
    k = np.minimum(k, len(x))
    ind = np.argpartition(x, -k)[-k:]
    return ind[x[ind].argsort()]


def top10(filename):
    date = date_from_filename(os.path.basename(filename))
    v, cols = data(filename)
    argtop, argbottom = argtopk(10, v), argtopk(10, -v)
    assert len(argtop) == len(argbottom), 'length of top and bottom not equal'
    return [(date, int(p // cols), p % cols, v[p])
            for p in chain(argtop, argbottom)]


@do
def store(data, path, header):
    with open(path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(concat(data))


def bagit(files, total_size):
    bag = db.from_sequence(files)
    hc = store(bag.map(top10), 'csv/hotcold.csv',
               header=('date', 'cat', 'row', 'col', 'temp'))
    sm = store(bag.map(summarize), 'csv/summary.csv',
               header=('date', 'len', 'mean', 'median', 'std'))
    return hc, sm


def timeit(f, files, total_size):
    dsk = do(tuple)(f(files, total_size))
    with ProgressBar():
        start = time()
        result = dsk.compute()
        stop = time()
    return result, stop - start


def mean_median_plot(csv):
    df = pd.read_csv(csv, header=0, parse_dates=['date'],
                     infer_datetime_format=True,
                     usecols=['date', 'mean', 'median'])
    output_file('mean_median.html')
    show(TimeSeries(df.sort('date'), title='Air Temperature',
                    index='date',
                    xlabel='Date', ylabel='Air Temperature (°C)', legend=True))


def heatmap(csv):
    import seaborn as sns
    from matplotlib import pyplot as plt
    df = pd.read_csv(csv, header=0, parse_dates=['date'],
                     infer_datetime_format=True)
    piv = (df.groupby(['row', 'col'], as_index=False)
             .temp
             .mean()
             .pivot('row', 'col', 'temp')
             .rename(columns=dict(row='Latitude', col='Longitude',
                                  temp='Average Temperature')))
    ax = sns.heatmap(piv)
    for i, (xlabel, ylabel) in enumerate(zip(ax.xaxis.get_ticklabels(),
                                             ax.yaxis.get_ticklabels())):
        should_show = bool(i % 50)
        xlabel.set_visible(should_show)
        ylabel.set_visible(should_show)

    plt.show()


def fileset_size(files):
    def get_h5py_size(path):
        with h5py.File(path, mode='r') as f:
            dset = f['/HDFEOS/GRIDS/NCEP/Data Fields/Tair_2m']
            return dset.size * dset.dtype.itemsize

    return sum(get_h5py_size(f) for f in files)


if __name__ == '__main__':
    files = [f for f in glob.glob(os.path.join('raw', '*.he5'))]
    total_size = fileset_size(files)
    _, d = timeit(bagit, files, total_size)
    print('bagit, %.2f G, %d files, %.2f s' % (total_size / 1e9, len(files), d))
    # heatmap('csv/hotcold.csv')
