from nltk.tokenize import word_tokenize


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features
