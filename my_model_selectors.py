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

        best_model = None
        best_score = float('inf')

        for number_of_states in range(self.min_n_components, self.max_n_components+1):
            try:
                current_model = self.base_model(number_of_states)
                curr_score = current_model.score(self.X, self.lengths)

                # number of params
                p = number_of_states**2 + 2*number_of_states*len(self.lengths) - 1

                bic = -2*curr_score + p*np.log(len(self.lengths))

                if bic < best_score:
                    best_score = bic
                    best_model = current_model
            except:
                pass    # ignore failed iterations

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

        best_model = None
        best_score = -float('inf')

        for number_of_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                current_model = self.base_model(number_of_states)
                curr_score = current_model.score(self.X, self.lengths)
                score_sum = 0

                for word in self.hwords:
                    if word != self.this_word:
                        x_w, l_w = self.hwords[word]
                        score_sum += current_model.score(x_w, l_w)

                # DIC = log(P(X(i)) - 1 / (M - 1)SUM(log(P(X(all but i))
                dic = curr_score - (1 / (len(self.hwords)-1) * score_sum)

                if dic > best_score:
                    best_score = dic
                    best_model = current_model
            except:
                pass  # ignore failed iterations

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = -float('inf')
        best_model = None

        for number_of_states in range(self.min_n_components, self.max_n_components+1):
            try:
                current_model = self.base_model(number_of_states)
                folds = KFold().split(self.sequences)
                current_score = 0
                count = 0
                for train, test in folds:
                    train_x, train_lengths = combine_sequences(train, self.sequences)
                    test_x, test_lengths = combine_sequences(test, self.sequences)

                    current_score += current_model.score(test_x, test_lengths)
                    count += 1
            except:
                pass    # ignore failed iterations
            count = 1 if count == 0 else count
            average_score = current_score / count

            if average_score > best_score:
                best_model = current_model
                best_score = average_score
        return best_model
