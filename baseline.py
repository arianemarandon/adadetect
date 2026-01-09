import numpy as np
from scipy.stats import chi2

from .algo import BH

class AgnosticBH(object):
    """
    Chi-square test. 
    """
    def __init__(self):
        pass
    def apply(self, x, level, xnull=None):
        """
        x: test sample 
        level: nominal level 

        Return: rejection set for the chi-square test applied to the test sample x. 
        """
        dimensionSize = x.shape[1]

        test_statistic = np.power(np.linalg.norm(x, axis=1),2) 
        pvalues = 1- chi2.cdf(test_statistic, df=dimensionSize) 

        return BH(pvalues, level)


class PlugInBH(object):
    """
    Local-fdr procedure of Sun and Cai. The estimator of the null density may either 
    take as input the test sample <x> to use for fitting (as in the original paper) 
    or take as input an additional NTS <xnull> to use for fitting (setting considered in our paper), see below. 
    """

    def __init__(self, scoring_fn_mixture, scoring_fn_null):
        """
        scoring_fn_mixture: A class (estimator) that must have a .fit() and a .score_samples() method, e.g. sklearn's KernelDensity() 
                            The .fit() method takes as input a (training) data sample and may set/modify some parameters of scoring_fn_mixture
                            The .score_samples() method takes as input a (test) data sample and should return the log-density for each element, as in sklearn's KernelDensity() 
        scorinf_fn_null: Same as above. 
        """
        self.scoring_fn_mixture = scoring_fn_mixture #estimator for the mixture density
        self.scoring_fn_null = scoring_fn_null #estimator for the null density

    def fit(self, x, level, xnull=None):
        self.scoring_fn_mixture.fit(x)
        if xnull is not None:
            self.scoring_fn_null.fit(xnull)
        else:
            self.scoring_fn_null.fit(x)

    def apply(self, x, level, xnull=None):
        """
        x: test sample 
        xnull: NTS (optional)
        level: nominal level 

        Return: rejection set 
        """
        self.fit(x, level, xnull)

        local_fdr_statistics = np.exp(self.scoring_fn_null.score_samples(x) - self.scoring_fn_mixture.score_samples(x))
        
        #Algorithm of Sun and Cai
        n = len(x)
        indices = np.argsort(local_fdr_statistics)
        Tsort = np.sort(local_fdr_statistics)

        Tsum = np.cumsum(Tsort) / np.arange(1, n + 1)

        if np.nonzero(Tsum < level)[0].size:
            n_sel = np.nonzero(Tsum < level)[0][-1] +1 
            rejection_set = indices[:n_sel]
        else: 
            rejection_set = np.array([])
        return rejection_set
