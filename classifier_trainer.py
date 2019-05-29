import nltk
import random
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from find_features import find_features
from speech import Speech


class NoDataError(Exception):
    pass


class TrainingData:
    def __init__(
            self,
            positive_list,
            negative_list,
            train_number=14000,
            feature_number=4000
    ):
        self.positive_list = positive_list
        self.negative_list = negative_list
        self.train_number = train_number
        self.feature_number = feature_number
        self.documents_list = []
        self.all_words_list = []
        self.word_features = None
        self.feature_sets = None

    def _make_documents(self):
        for document in self.positive_list.split('\n'):
            # optimized_document = Speech(document)
            self.documents_list.append(
                (
                    document,
                    "pos"
                )
            )

        for document in self.negative_list.split('\n'):
            # optimized_document = Speech(document)
            self.documents_list.append(
                (
                    document,
                    "neg"
                )
            )

    def _append_all_words(self):
        short_pos_words = self.positive_list
        short_neg_words = self.negative_list

        for word in nltk.word_tokenize(short_pos_words):
            self.all_words_list.append(word)

        for word in nltk.word_tokenize(short_neg_words):
            self.all_words_list.append(word)

    def prepare_data(self):
        self._make_documents()
        self._append_all_words()

        self.word_features = list(
            nltk.FreqDist(self.all_words_list).keys()
        )[:self.feature_number]
        self.feature_sets = [
            (
                find_features(rev, self.word_features),
                category
            ) for (rev, category) in self.documents_list
        ]
        random.shuffle(self.feature_sets)

    def get_testing_set(self):
        if not self.feature_sets:
            self.prepare_data()
        return self.feature_sets[self.train_number:]

    def get_training_set(self):
        if not self.feature_sets:
            self.prepare_data()
        return self.feature_sets[:self.train_number]

    def save_word_features(self):
        if self.word_features:
            with open('word_features.pickle', 'wb') as file:
                pickle.dump(self.word_features, file)
        else:
            raise NoDataError


class ClassificationSet:
    def __init__(self, **kwargs):
        self.classifiers = kwargs

    def save_set(self):
        for key, classifier in self.classifiers.items():
            with open(f'{key}.pickle', 'wb') as file:
                pickle.dump(classifier, file)


def main():
    with open("positive.txt", "r") as file_positive,\
            open("negative.txt", "r") as file_negative:
        positive_list = file_positive.read()
        negative_list = file_negative.read()

    data = TrainingData(positive_list, negative_list)
    training_set = data.get_training_set()
    testing_set = data.get_testing_set()

    mnb_classifier = SklearnClassifier(MultinomialNB())
    mnb_classifier.train(training_set)
    print("mnb_classifier accuracy percent:",
          (nltk.classify.accuracy(mnb_classifier, testing_set)) * 100)

    logistic_regression_classifier = SklearnClassifier(
        LogisticRegression(solver='lbfgs')
    )
    logistic_regression_classifier.train(training_set)
    print("logistic_regression_classifier accuracy percent:", (
        nltk.classify.accuracy(
            logistic_regression_classifier,
            testing_set
        )
    ) * 100)

    linearsvc_classifier = SklearnClassifier(LinearSVC())
    linearsvc_classifier.train(training_set)
    print("linearsvc_classifier accuracy percent:",
          (nltk.classify.accuracy(linearsvc_classifier, testing_set)) * 100)

    classifiers_set = ClassificationSet(
        mnb_classifier=mnb_classifier,
        logistic_regression_classifier=logistic_regression_classifier,
        linearsvc_classifier=linearsvc_classifier,
    )

    classifiers_set.save_set()
    data.save_word_features()


if __name__ == '__main__':
    main()
