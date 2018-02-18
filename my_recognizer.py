import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for i, (x, x_lengths) in test_set.get_all_Xlengths().items():
        all_scores = {}
        best_score = -float('inf')
        best_word = ""
        for w, m in models.items():
            try:
                # print(m)
                current_score = m.score(x, x_lengths)
                if current_score > best_score:
                    best_score = current_score
                    best_word = w
                all_scores.setdefault(w, current_score)
                # print(current_score)
            except:
                pass
        probabilities.append(all_scores)
        guesses.append(best_word)
    # return probabilities, guesses
    return probabilities, guesses
