import numpy as np
import matplotlib.pyplot as plt


class TestResult:
    def __init__(self, windows, stats, crits):
        self.windowStatCrit = list(zip(windows, stats, crits))

    def isRejected(self):
        """ returns if Hnull rejected

        :return: bool
        """
        return any([np.max(stat) > crit for (window, stat, crit) in self.windowStatCrit])

    def getPointEstimate(self):
        """ returns location maximizing the test statistic

        :return: There is no Pythonic Maybe monad, so we return either an empty list (if Hnull is not rejected),
        or a list of a single element
        """
        if self.isRejected():
            (window, stats, _) = max(self.windowStatCrit, key=lambda wsc: wsc[0])
            return [np.argmax(stats) + window]
        else:
            return []

    def getLocalisingWindow(self):
        """ returns the narrowes window rejecting Hnull

        :return: There is no Pythonic Maybe monad, so we return either an empty list (if Hnull is not rejected),
        or a list of a single element
        """
        if self.isRejected():
            return [min([window for (window, stat, crit) in self.windowStatCrit if max(stat) > crit])]
        else:
            return []

    def plot(self):
        nPlots = len(self.windowStatCrit)
        f, subplots = plt.subplots(nPlots, 1, sharex='col')
        if nPlots == 1:
            subplots = [subplots]
        for (i, (subplot, (window, stats, crit))) in enumerate(zip(subplots, self.windowStatCrit)):
            time = np.arange(window, window + len(stats))
            subplot.plot(time, stats)
            subplot.axhline(y=crit)
            subplot.set_title("Window size = %d" % window)
        plt.show()

    def cutoff(self):
        return np.min([np.where(stat > crit)[0] + 2 * window if np.any(stat > crit) else stat.shape[0] + window for (window, stat, crit) in self.windowStatCrit])

    def partialMax(self):
        if self.isRejected():
            (window, stats, _) = max(self.windowStatCrit, key=lambda wsc: wsc[0])
            return [np.argmax(stats[:self.cutoff()]) + window]
        else:
            return []
