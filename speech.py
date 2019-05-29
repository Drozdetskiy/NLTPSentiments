import re
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.tokenize import word_tokenize


class Speech:
    def __init__(
            self,
            contest,
            raw_patten=r'[а-я]',
            stemmer=SnowballStemmer('russian'),
            stop_words=stopwords.words('russian')
    ):

        self.contest = contest
        self.pattern = re.compile(raw_patten, re.IGNORECASE)
        self.stemmer = stemmer
        self.stop_words_set = set(stop_words)
        self._optimized_words = []

    def optimize(self):
        _words = word_tokenize(self.contest)
        for word in _words:
            result = self.pattern.match(word)
            if result and word not in self.stop_words_set:
                self._optimized_words.append(self.stemmer.stem(word.lower()))

    def get_optimized_words(self):
        if not self._optimized_words:
            self.optimize()
        return self._optimized_words

    def get_optimized_contest(self):
        return ' '.join(self.get_optimized_words())
