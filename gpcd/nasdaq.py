import gpcd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.interactive(False)
data = pd.read_csv("~/nasdaq.csv")
close = data[['Close']]
date = data[['Date']].values

X = close.values
lr = np.log(X[1:, :] / X[:-1, :])
X = X[1:, :]
dX = close.diff().values[1:, :]
plt.show()
prepared = np.concatenate([X, dX / X], axis=1)
window = 20

# ['1998-08-31']  Computer virus CIH activates and attacks Windows 9x.
# ['2001-04-03']  .com bubble
# ['2008-11-20']  global financial crysis


def detectMultiple():
    global result, cutoff, prepared, date
    while prepared.shape[0] > 300:
        result = gpcd.doTheTest(prepared, 300, (window,), 1000)
        cutoff = result.cutoff()
        print(date[cutoff - window])
        print(cutoff - window)
        result.plot()
        prepared = prepared[cutoff:, :]
        date = date[cutoff:, :]


detectMultiple()
prepared = prepared[3400:, :]
date = date[3400:, :]
detectMultiple()



