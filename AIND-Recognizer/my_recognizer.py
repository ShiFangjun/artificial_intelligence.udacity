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
    # TODO implement the recognizer
    for i in range(test_set.num_items):
        best_prob = None
        best_word = None
        temp_word_probability = {}
        temp_sequences, temp_lengths = test_set.get_item_Xlengths(i)
        for word, model in models.items():
            try:
                temp_word_probability[word] = model.score(temp_sequences, temp_lengths)
            except:
                temp_word_probability[word] = None
            if(best_prob == None or temp_word_probability[word] == None or temp_word_probability[word] > best_prob):
                best_prob, best_word = temp_word_probability[word], word
            continue
        probabilities.append(temp_word_probability)
        guesses.append(best_word)
    return probabilities, guesses
