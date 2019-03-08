import gpcd

import numpy as np
from numpy.random import normal
from functools import partial
from joblib import Parallel, delayed
import statsmodels.stats.proportion
import time
import sys
import matplotlib.pyplot as plt


plt.interactive(False)


def runTrial(phi, burnInSize, windowSizes, bootIter):
    data = generateDataset(phi)
    return gpcd.doTheTest(data, burnInSize, windowSizes, bootIter)


def generateDataset(phi):
    n = 800
    nAfterBreak = 100
    X = np.linspace(0, np.pi, num=n).reshape(-1, 1)
    y = np.sin(X) + normal(0, 0.1, size=n).reshape(-1, 1)
    data = np.concatenate((X, y), axis=1)
    np.random.shuffle(data)
    data[-nAfterBreak:, -1] = (np.sin(data[-nAfterBreak:, 0] + phi) + normal(0, 0.1, size=nAfterBreak))
    return data


def breakExtent(phi):
    X = np.linspace(0, np.pi, num=1000).reshape(-1, 1)
    return np.mean(np.abs(np.sin(X) - np.sin(X + phi)))


def runExperiment(runTrialSpecified, phi, iters, jobs):
    def runner(i):
        return runTrialSpecified(phi=phi)

    startTime = time.time()
    results = Parallel(n_jobs=jobs)(delayed(runner)(i) for i in range(iters))

    print("BREAK \t" + str(breakExtent(phi)))

    numOfRejections = sum([result.isRejected() for result in results])
    print("POWER\t" + str(numOfRejections / iters) + " in " + str(wilsonConfInterval(iters, numOfRejections)))

    trueLocation = 700
    sd = np.sqrt((np.array([p - trueLocation for result in results for p in result.getPointEstimate()])**2).mean())
    print("STDEV\t" + str(sd))

    localization = np.array([w for result in results for w in result.getLocalisingWindow()]).mean()
    print("LOC\t" + str(localization))
    print("TIME\t" + str(time.time() - startTime))
    sys.stdout.flush()


def wilsonConfInterval(nTrials, nSuccsesull):
    return statsmodels.stats.proportion.proportion_confint(nSuccsesull, nTrials, method='wilson')


def experiment(piFraction, windowSizes):
    [print("===========================") for _ in range(1, 5)]
    print("WINDOW\t" + str(windowSizes))
    print("PHI\tpi/" + str(piFraction))
    phi = np.pi / piFraction
    runTrialSpecified = partial(runTrial, bootIter=1000, burnInSize=500, windowSizes=windowSizes)
    runExperiment(runTrialSpecified, phi, 800, 80)


[experiment(np.Inf, (w,)) for w in [40, 20, 10, 5]]
[experiment(piFraction, (40,)) for piFraction in [10, 20, 40]]
[experiment(piFraction, (20,)) for piFraction in [5, 10, 20, 40]]
[experiment(piFraction, (10,)) for piFraction in [5, 10, 20, 40]]
[experiment(piFraction, (5,)) for piFraction in [2, 5, 10, 20]]
experiment(np.Inf, (40, 20))
experiment(np.Inf, (40, 20, 10))
experiment(np.Inf, (40, 20, 10, 5))
experiment(10, (40, 20))
experiment(10, (40, 20, 10))
experiment(10, (40, 20, 10, 5))
# runTrialSpecified = partial(runTrial, bootIter=100, burnInSize=500, windowSizes=[10])
# runExperiment(runTrialSpeci   fied, np.pi, 80, 80)
# res = runTrialSpecified(phi=3.14)
# print(res.getPointEstimate())
# print(res.getLocalisingWindow())
# print(res.isRejected())
# res.plot()
