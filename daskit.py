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
import shutil

from itertools import chain
from time import time

import h5py

import numpy as np

import dask.bag as db

from dask import do, value
from dask.diagnostics import ProgressBar

from cytoolz import concat

from pyspark import SparkContext

sc = SparkContext()

rx = re.compile(r'^GSSTF_NCEP\.3\.(\d{4}\.\d{2}\.\d{2})\.he5$')


def data(filename):
    with h5py.File(filename, mode='r') as f:
        dset = f['/HDFEOS/GRIDS/NCEP/Data Fields/Tair_2m']
        fill = dset.attrs['_FillValue'][0]
        x = dset[:].ravel()
        cols = dset.shape[1]
    cond = x != fill
    nmissing = len(x) - cond.sum()
    assert nmissing > 0
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


def sparkit(files):
    bag = sc.parallelize(files).map(data)
    paths = tensf, sparkf = 'csv/hotcold.spark.csv', 'csv/summary.spark.csv'
    for p in paths:
        if os.path.exists(p):
            shutil.rmtree(p)
    tens = bag.map(lambda args: ','.join(map(str, top10(*args)[0]))).saveAsTextFile(tensf)
    summaries = bag.map(lambda args: ','.join(map(str, summarize(*args)[0]))).saveAsTextFile(sparkf)
    return tens, summaries


def timeit(f, files):
    if f.__name__ == 'sparkit':
        start = time()
        result = f(files)
        stop = time()
        return result, stop - start
    else:
        dsk = value(f(files))
        start = time()
        with ProgressBar():
            result = dsk.compute()
        stop = time()
        return result, stop - start


def fileset_size(files):
    def get_h5py_size(path):
        with h5py.File(path, mode='r') as f:
            dset = f['/HDFEOS/GRIDS/NCEP/Data Fields/Tair_2m']
            return dset.size * dset.dtype.itemsize

    return sum(get_h5py_size(f) for f in files)


if __name__ == '__main__':
    files = [f for f in glob.glob(os.path.join('raw', '*.he5'))]
    total_size = fileset_size(files)
    runner = dict(do=doit, bag=bagit, spark=sparkit)[sys.argv[1]]
    _, d = timeit(runner, files)
    print('%s, %.2f G, %d files, %.2f s' %
          (runner.__name__, total_size / 1e9, len(files), d))
    # heatmap('csv/hotcold.csv')
