import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        best_score = float('inf')
        best_model = None
        logN = np.log10(sum(self.lengths))
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                p = n_components + n_components * (n_components - 1) + 2 * n_components * self.X.shape[1] 
                BIC = -2 * logL + p * logN
            except:
                continue
            if best_score > BIC:
                best_score = BIC
                best_model = model
        return best_model
        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = float('-inf')
        best_model = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            logL = {}
            valid_words = []
            for trained in (self.words).keys():
                X_train, lengths_train = self.hwords[trained]
                try:
                    model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    logL[trained] = model.score(X_train, lengths_train)
                    valid_words.append(trained)
                except:
                    pass
            try:
                DIC = logL[self.this_word] - (1 / (len(valid_words) - 1)) * sum([logL[valid] for valid in valid_words if valid != self.this_word])
                if DIC > best_score:
                    best_score = DIC
                    best_model = model
            except:
                best_model = model
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float('-inf')
        best_model = None
        split_method = KFold()
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            score = 0
            folds = 0
            try:
                if len(self.sequences) > 2:
                    for train_i, test_i in split_method.split(self.sequences):
                        train_X, train_l = combine_sequences(train_i, self.sequences)
                        test_X, test_l = combine_sequences(test_i, self.sequences)
                        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(train_X, train_l)
                        score += model.score(test_X, test_l)
                        folds += 1
                    avg_score = score / folds
                else:
                    model = self.base_model(n_components)
                    avg_score = model.score(self.X, self.lengths)
                if avg_score > best_score :
                    best_score = avg_score
                    best_model = model
            except:
                pass
        return best_model