#!/usr/bin/env python
# -*- coding: utf8 -*-


"""
Dask version of
https://hdfgroup.org/wp/2015/04/putting-some-spark-into-hdf-eos/
"""

from __future__ import print_function, division

import os
import sys
import glob
import re
import csv

from itertools import chain
from time import time

import h5py

import numpy as np
import pandas as pd

import dask.bag as db

from dask import do, value
from dask.diagnostics import ProgressBar

from cytoolz import concat
from toolz.compatibility import map, zip

from bokeh.charts import TimeSeries, show, output_file


rx = re.compile(r'^GSSTF_NCEP\.3\.(\d{4}\.\d{2}\.\d{2})\.he5$')


def data(filename):
    with h5py.File(filename, mode='r') as f:
        dset = f['/HDFEOS/GRIDS/NCEP/Data Fields/Tair_2m']
        fill = dset.attrs['_FillValue'][0]
        x = dset[:]
        cols = dset.shape[1]
    cond = x != fill
    nmissing = len(x) - cond.sum()
    date = re.match(rx, os.path.basename(filename)).group(1).replace('.', '-')
    return date, nmissing, x[cond], cols


def summarize(date, nmissing, v, cols):
    return [(date, len(v), np.mean(v), np.median(v), np.std(v, ddof=1))]


def argtopk(k, x):
    k = np.minimum(k, len(x))
    ind = np.argpartition(x, -k)[-k:]
    return ind[x[ind].argsort()]


def top10(date, nmissing, v, cols):
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


def bagit(files):
    bag = db.from_sequence(files).map(data)
    hc = store(bag.map(top10), 'csv/hotcold.csv',
               header=('date', 'cat', 'row', 'col', 'temp'))
    sm = store(bag.map(summarize), 'csv/summary.csv',
               header=('date', 'len', 'mean', 'median', 'std'))
    return hc, sm


def doit(files):
    datafiles = [do(data)(f) for f in files]
    tens = [do(top10)(v[0], v[1], v[2], v[3]) for v in datafiles]
    summaries = [do(summarize)(v[0], v[1], v[2], v[3])
                 for v in datafiles]
    hc = store(tens, 'csv/hotcold.csv',
               header=('date', 'cat', 'row', 'col', 'temp'))
    sm = store(summaries, 'csv/summary.csv',
               header=('date', 'len', 'mean', 'median', 'std'))
    return hc, sm


def timeit(f, files):
    dsk = value(f(files))
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
    runner = dict(do=doit, bag=bagit)[sys.argv[1]]
    _, d = timeit(runner, files)
    print('%s, %.2f G, %d files, %.2f s' %
          (runner.__name__, total_size / 1e9, len(files), d))
    # heatmap('csv/hotcold.csv')
