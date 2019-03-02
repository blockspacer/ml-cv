# coding: utf-8

import pandas as pd
import numpy as np
import hashlib
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

    # Empty dataframe
    df = pd.DataFrame(columns=['timeticket', 'hash_val', 'h_m'])

    # Append
    for i in range(16):
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

        df = df.append({'timeticket': timeticket, 'hash_val': hashlib.md5('black' + str(i)).hexdigest(),
                        'h_m': 5.0 + np.random.rand()},
                       ignore_index=True)

    print df

    # Append to hdf5

    # Read hdf5 by chunks
