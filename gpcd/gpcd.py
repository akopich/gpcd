from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
from numpy.linalg import inv


from TestResult import TestResult


def centeredNormalLogPDF(invCov, X):
    return -np.dot(X, np.dot(invCov, X.T))


# Last column is the response
def gpLogLikelihood(invKernel, data):
    return centeredNormalLogPDF(invKernel, data[:, -1].reshape(-1))


def fitKernel(data, gp):
    X = data[:, 0:-1]
    y = data[:, -1]
    gp.fit(X, y)
    yhat = gp.predict(X)
    eps = y - yhat
    return gp.kernel_, gp, eps


def getStatistic(invertedKernelWindow, invertedKernelLeft, invertedKernelRight, data):
    n = data.shape[0]
    left = data[0:n // 2, :]
    right = data[n // 2:n, :]
    return gpLogLikelihood(invertedKernelLeft, left) + gpLogLikelihood(invertedKernelRight, right) \
           -gpLogLikelihood(invertedKernelWindow, data)


def runningWindow(data, windowSize, fun):
    validStartIndxs = range(0, data.shape[0] - windowSize)
    return list(map(lambda start: fun(data[start:start + windowSize]), validStartIndxs))


def getInvertedKernels(data, kernel, windowSizes):
    return dict(map(lambda windowSize: (
    windowSize, runningWindow(data, windowSize, lambda windowData: inv(kernel(windowData[:, :-1])))), windowSizes))


def getStatistics(data, invertedKernels, windowSize):
    invertedKernelsForDoubleWindow = invertedKernels.get(windowSize * 2)
    invertedKernelsForWindow = invertedKernels.get(windowSize)
    stats = []
    for start in range(0, data.shape[0] - windowSize * 2):
        invkw = invertedKernelsForDoubleWindow[start]
        invkl = invertedKernelsForWindow[start]
        invkr = invertedKernelsForWindow[start + windowSize]
        stat = getStatistic(invkw, invkl, invkr, data[start:start + windowSize * 2])
        stats.append(stat)
    return np.array(stats)


def residualBootstrap(data, invKernels, windows, eps, gp, iters):
    data = data.copy()
    yhat = gp.predict(data[:, :-1])

    epsFixed = np.concatenate([eps, -eps])

    def bootIter():
        epsBootstrap = np.random.choice(epsFixed, data.shape[0])
        data[:, -1] = yhat + epsBootstrap
        return [max(getStatistics(data, invKernels, window)) for window in windows]

    stats = np.concatenate([np.array(bootIter()).reshape(1, -1) for _ in range(iters)], axis=0)
    return stats


def chanceOfRejection(stats, critLevels):
    res = np.any(stats > np.repeat(critLevels.reshape(1, -1), stats.shape[0], axis=0), axis=1)
    return np.sum(res) / res.shape[0]


def doTheTest(data, burnInN, windows, bootIter):
    """ Computes the statistics, runs the bootstrap and produces the TestResult object

    :param data: numpy 2D array, the last column should contain responses.
    :param burnInN: the number of the first data point to be used for bootstrap
    :param windows: list or tuple of window sizes
    :param bootIter:
    :return: TestResult
    """
    burnInData = data[:burnInN, :]
    kernel, gp, eps = fitKernel(burnInData, GP(kernel=1 * RBF() + WhiteKernel()))
    windowAndDoubledSizes = sorted(list(set([windowOrDoubled for w in windows for windowOrDoubled in [w, w * 2]])))
    invKernels = getInvertedKernels(data, kernel, windowAndDoubledSizes)
    stats = [getStatistics(data, invKernels, window) for window in windows]
    bootStats = residualBootstrap(data, invKernels, windows, eps, gp, bootIter)
    crits = multiplicityCorrectedCrits(bootStats, 0.01)
    return TestResult(windows, stats, crits)


def multiplicityCorrectedCrits(bootStats, alpha):
    sorted = np.sort(bootStats, axis=0)
    sorted = sorted[::-1, :]
    i = 0
    while chanceOfRejection(bootStats, sorted[i,]) < alpha:
        i = i + 1
    i = max(0, i - 1)
    return sorted[i, ].tolist()


