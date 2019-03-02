# coding: utf-8

import pandas as pd
import numpy as np
import hashlib
from pandas import HDFStore
import md5

from datetime import datetime
from datetime import timedelta

if __name__ == '__main__':
    # # Хранить в hdf5, но обрабатывать кусками
    # # append?
    # # https://stackoverflow.com/questions/17098654/how-to-store-a-dataframe-using-pandas
    # # 'C': pd.Series(1, index=list(range(4)), dtype='float32'),  # как то задает количество строк
    # df2 = pd.DataFrame({
    #     'D': np.array([3] * 4, dtype='int32'),
    #     'F': 'foo'})
    #
    # print df2

    #  index=dates,
    # df2 = pd.DataFrame(index=df1.index.copy())
    # df = pd.DataFrame(np.random.randn(0, 0), columns=list('ABCD'))
    # print df


    # https://github.com/pandas-dev/pandas/issues/14655
    # Empty dataframe
    # https://pbpython.com/pandas_dtypes.html

    dtypes = {'timeticket': 'datetime64[ns]',
              'hash_val': 'object',  # str the same?
              'h_m': 'float64'}
    columns = ['timeticket', 'hash_val', 'h_m']

    # Append
    # FIXME: faster!
    # https://stackoverflow.com/questions/22998859/hdfstore-with-string-columns-gives-issues
    vec = []
    for i in range(500):
        # https://stackoverflow.com/questions/44513738/pandas-create-empty-dataframe-with-only-column-names/44514187
        # timeticket = datetime.now()
        # timeticket = datetime.now()
        # print timeticket.time
        # 1551189341.122005
        sec = 1551189341 + i
        usec = 22005 + np.random.randint(0, 128)
        # https://stackoverflow.com/questions/3682748/converting-unix-timestamp-string-to-readable-date
        s_dt = datetime.fromtimestamp(sec)
        us_dt = timedelta(microseconds=usec)  # datetime.fromtimestamp(usec)
        timeticket = s_dt + us_dt
        # print datetime.datetime.combine(s_dt, us_dt)

        # FIXME: Recreation! bad
        row = {'timeticket': timeticket, 'hash_val': hashlib.md5('black' + str(i)).hexdigest(),
               'h_m': 5.0 + np.random.rand()}
        vec.append(row)

        # Very slow
        # df = df.append(},
        #                ignore_index=True)

    df = pd.DataFrame(vec, columns=columns)  # )
    for c in df.columns:
        df[c] = df[c].astype(dtypes[c])

    print df.info()
    # print df.dtypes
    print 'Store...'
    # print df

    # Append to hdf5 - no it's better to recreate!
    fn = '/tmp/data.h5'
    store = HDFStore(fn)
    # adding dataframe to the HDF5 file
    # store.put('d1', df, format='table', data_columns=True)

    # , data_columns=True
    # store['df'] = df  # save it
    # store.put('d1', df, data_columns=True)  # no chunks
    store.put('df', df, data_columns=True, format='table')
    print df

    store.close()
    # store['df']  # load it
    #
    # Read hdf5 by chunks
    # https://towardsdatascience.com/why-and-how-to-use-pandas-with-large-data-9594dda2ea4c
    # https://stackoverflow.com/questions/40348945/reading-data-by-chunking-with-hdf5-and-pandas
    rd_store = HDFStore(fn)
    #
    # df = pd.DataFrame(columns=columns)
    # chunksize = 4096
    # # %%timeit
    # # for chunk in pd.read_hdf(fn, 'df', chunksize=chunksize, where='h_m < 5.3'):
    # #     df = pd.concat([df, chunk], ignore_index=True)
    #
    # # sel by time
    # # https://stackoverflow.com/questions/25681308/pandas-read-hdf-query-by-date-and-time-range
    # # Может лучше не таблицей хранить если выбирать по времени
    c = rd_store.select_column('df', 'timeticket')

    where = pd.DatetimeIndex(c).indexer_between_time('12:00', '16:56')
    #
    resp = rd_store.select('df', where=where)
    # print resp.info()
    print resp

    # Another selection
    # https://stackoverflow.com/questions/20502996/use-or-in-hdfstore-select-pandas
