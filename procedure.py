import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import GridSearchCV, ParameterGrid
from functools import reduce

from algo import BH, EmpBH, adaptiveEmpBH
    


#---------------------------------------------------Baselines (previous work): chi-square test and local-fdr procedure

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



#---------------------------------------------------AdaDetect (ours)
class AdaDetectBase(object):
    """
    Base template for AdaDetect procedures to inherit from. 
    """

    def __init__(self, correction_type=None, storey_threshold=0.5):
        """
        correction_type: if 'storey'/'quantile', uses the adaptive AdaDetect procedure with storey/quantile correction
        """
        self.null_statistics = None
        self.test_statistics = None 
        self.correction_type = correction_type
        self.storey_threshold = storey_threshold

    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics>. 
        """
        #This part depends specifically on the type of AdaDetect procedure: 
        #whether the scoring function g is learned via density estimation, or an ERM approach (PU classification)
        #Thus, it is coded in separate AdaDetectBase objects, see below. 

        pass
    
    def apply(self, x, level, xnull): 
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: rejection set of AdaDetect with scoring function g learned from <x> and <xnull> as per .fit(). 
        """ 
        self.fit(x, level, xnull)
        if self.correction_type is not None:
            return adaptiveEmpBH(self.null_statistics, self.test_statistics, level = level, 
            correction_type = self.correction_type, storey_threshold = self.storey_threshold)
        else: 
            return EmpBH(self.null_statistics, self.test_statistics, level = level)


class AdaDetectDE(AdaDetectBase):
    """
    AdaDetect procedure where the scoring function is learned by a density estimation approach. There are two possibilities: 
        - Either the null distribution is assumed known, in which case the scoring function is learned on the mixed sample = test sample + NTS. 
        - Otherwise, the NTS is split, and the scoring function is learned separatly on a part of the NTS (to learn the null distribution) and on the remaining mixed sample. 

    Note: one-class classification (approach Bates et. al) can be obtained from this routine: it suffices to define scoring_fn (see below) such that only the first part of the NTS is used. 
    """

    def __init__(self, scoring_fn, f0_known=True, split_size=0.5, correction_type=None, storey_threshold = 0.5):
        AdaDetectBase.__init__(self, correction_type, storey_threshold)
        """
        scoring_fn: A class (estimator) that must have a .fit() and a .score_samples() method, e.g. sklearn's KernelDensity() 
                            The .fit() method takes as input a (training) data sample and may set/modify some parameters of scoring_fn
                            The .score_samples() method takes as input a (test) data sample and should return the log-density for each element, as in sklearn's KernelDensity() 
        The same method is used for learning the null distribution as for the 'mixture distribution' of the test sample mixed with the second part of the NTS ('f_gamma' in the paper). 

        f0_known: boolean, indicates whether the null distribution is assumed known (=True, in that case scoring_fn should use this knowledge, 
        e.g. by returning in its score_samples() method the ratio of a fitted mixture density estimator over the true null density) or not (=False)

        split_size: proportion of the part of the NTS used for fitting g i.e. k/n with the notations of the paper
        """
        self.scoring_fn = scoring_fn
        self.f0_known = f0_known
        self.split_size = split_size
        
    
    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics> properties (which are properties of any AdaDetectBase object) 
        """
        m = len(x)
        n = len(xnull)

        # learn the scoring function
        if self.f0_known: 

            x_train = np.concatenate([xnull, x]) 
        
            self.scoring_fn.fit(x_train)

        else:

            #split the null
            n_null_train = int(self.split_size * n)
            xnull_train = xnull[:n_null_train] #this is set aside for learning the score
            xnull_calib = xnull[n_null_train:] #must NOT be set aside!!! must be mixed in with x to keep control 

            xtrain = np.concatenate([xnull_calib, x])

            self.scoring_fn.fit(x_train = xtrain, x_null_train = xnull_train)

            xnull = xnull_calib

        # compute scores 
        self.test_statistics = self.scoring_fn.score_samples(x) 
        self.null_statistics = self.scoring_fn.score_samples(xnull) 


class AdaDetectERM(AdaDetectBase):
    """
    AdaDetect procedure where the scoring function is learned by an ERM approach. 
    """


    def __init__(self, scoring_fn, split_size=0.5, correction_type=None, storey_threshold=0.5):
        AdaDetectBase.__init__(self, correction_type, storey_threshold)
        """
        scoring_fn: A class (estimator) that must have a .fit() and a .predict_proba() or .decision_function() method, e.g. sklearn's LogisticRegression() 
                            The .fit() method takes as input a (training) data sample of observations AND labels <x_train, y_train> and may set/modify some parameters of scoring_fn
                            The .predict_proba() method takes as input a (test) data sample and should return the a posteriori class probabilities (estimates) for each element
        
        split_size: proportion of the part of the NTS used for fitting g i.e. k/n with the notations of the paper
        """

        self.scoring_fn = scoring_fn
        self.split_size = split_size

    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics> properties (which are properties of any AdaDetectBase object) 
        """
        m = len(x)
        n = len(xnull)

        n_null_train = int(self.split_size * n) 
        xnull_train = xnull[:n_null_train]
        xnull_calib = xnull[n_null_train:]

        x_mix_train = np.concatenate([x, xnull_calib])

        #fit a classifier using xnull_train and x_mix_train
        x_train = np.concatenate([xnull_train, x_mix_train])
        y_train = np.concatenate([np.zeros(len(xnull_train)), np.ones(len(x_mix_train))])
        
        self.scoring_fn.fit(x_train, y_train)

        # compute scores 
        methods_list = ["predict_proba", "decision_function"]
        prediction_method = [getattr(self.scoring_fn, method, None) for method in methods_list]
        prediction_method = reduce(lambda x, y: x or y, prediction_method)

        self.null_statistics = prediction_method(xnull_calib)
        self.test_statistics = prediction_method(x)

        if self.null_statistics.ndim != 1:
            self.null_statistics = self.null_statistics[:,1]
            self.test_statistics = self.test_statistics[:,1]



class AdaDetectERMcv(AdaDetectBase): 
    """
    AdaDetect procedure where the scoring function is learned by an ERM approach, with cross-validation scheme of the paper.  
    """
    def __init__(self, scoring_fn, cv_params=None, split_size=0.5):
        """
        scoring_fn: A class (estimator) that must have a .fit() and a .predict_proba() or .decision_function() method as in 'AdaDetectERM'
                    Additionally, must have a .set_params() method that takes as input a dictionary with keys being parameter names and values being parameter values 
        
        cv_params: A dictionary with keys being parameter names (as named in <scoring_fn> class) and values being a list of parameter values
                   For instance: scoring_fn = RandomForest(), cv_params = {'max_depth': [3, 5, 10]}


        split_size: this is k/n using the notations of the paper. (The second split is done such that k-s=l+m i.e. s = k-(l+m) as per the recommandations for choosing s in our paper.)
        """
        AdaDetectBase.__init__(self)
        self.scoring_fn = scoring_fn
        self.default_scoring_fn_params = scoring_fn.get_params() 
        self.cv_params = cv_params 
        self.split_size = split_size

    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics> properties (which are properties of any AdaDetectBase object) 
        """
        m = len(x)
        n = len(xnull)
        
        if self.cv_params is not None:
            n_null_train = int(self.split_size * n) #k 
            n_calib = n-n_null_train #l
            n_calib_2 = n_calib
            n_calib_1 = n_calib_2 + m #k-s = l+m

            xnull_train = xnull[:n_null_train] #Y_1, ..., Y_k
            xnull_calib_1 = xnull_train[:n_calib_1] #Y_(s+1), ..., Y_k
            xnull_train = xnull_train[n_calib_1:] #Y_1, ..., Y_s
            xnull_calib_2 = xnull[n_null_train:] #Z_(k+1), ..., Z_(n+m)

            new_x = np.concatenate([x, xnull_calib_2])

            new_x_null = np.concatenate([xnull_train, xnull_calib_1])
            grid = list(ParameterGrid(self.cv_params))
            max_power=0
            best_params=None

            split_size = len(xnull_train) / len(new_x_null)

            for parameter_cb in grid:
                self.scoring_fn.set_params(**parameter_cb)
                rejection_set = AdaDetectERM(scoring_fn = self.scoring_fn, split_size = split_size).apply(x=new_x, level=level, xnull=new_x_null)
                
                power = len(rejection_set)
                if power > max_power: 
                    best_params = parameter_cb
                    max_power = power 
                
            if max_power==0:
                #then choose default params
                self.scoring_fn = self.scoring_fn.set_params(**self.default_scoring_fn_params)
            else:
                self.scoring_fn.set_params(**best_params)
            #the outcome is a function of (new_x_null, new_x)

            xnull_train = new_x_null
            x_mix_train = new_x     

            x_train = np.concatenate([xnull_train, x_mix_train])
            y_train = np.concatenate([np.zeros(len(xnull_train)), np.ones(len(x_mix_train))])
            #now fit scoring_fn 
            self.scoring_fn.fit(x_train, y_train)
            self.null_statistics = self.scoring_fn.predict_proba(xnull_calib_2)[:,1]
            self.test_statistics = self.scoring_fn.predict_proba(x)[:,1]
            
        else:
            proc = AdaDetectERM(scoring_fn= self.scoring_fn)
            proc.fit(x=x, level=level, xnull=xnull)
            self.null_statistics = proc.null_statistics
            self.test_statistics= proc.test_statistics
            
        



