import pickle

from find_features import find_features
from vote_classifier import VoteClassifier


def load_classifiers(*args):
    classifiers = []
    for classifier in args:
        with open(f'{classifier}.pickle', 'rb') as file:
            classifiers.append(pickle.load(file))
    return classifiers


def sentiment(text, word_features, voted_classifier):
    feats = find_features(text, word_features)
    return voted_classifier.classify(feats), \
        voted_classifier.confidence(feats)


def main(text):
    classifiers = load_classifiers(
        'mnb_classifier',
        'logistic_regression_classifier',
        'linearsvc_classifier',
    )
    with open('word_features.pickle', 'rb') as file:
        word_features = pickle.load(file)

    voted_classifier = VoteClassifier(*classifiers)

    return sentiment(text, word_features, voted_classifier)
