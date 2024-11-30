# Natural Language Toolkit: Interface to scikit-learn classifiers
#
# Author: Lars Buitinck <L.J.Buitinck@uva.nl>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
scikit-learn (https://scikit-learn.org) is a machine learning library for
Python. It supports many classification algorithms, including SVMs,
Naive Bayes, logistic regression (MaxEnt) and decision trees.

This package implements a wrapper around scikit-learn classifiers. To use this
wrapper, construct a scikit-learn estimator object, then use that to construct
a SklearnClassifier. E.g., to wrap a linear SVM with default settings:

    >>> from sklearn.svm import LinearSVC
    >>> from nltk.classify.scikitlearn import SklearnClassifier
    >>> classif = SklearnClassifier(LinearSVC())

A scikit-learn classifier may include preprocessing steps when it's wrapped
in a Pipeline object. The following constructs and wraps a Naive Bayes text
classifier with tf-idf weighting and chi-square feature selection to get the
best 1000 features:

    >>> from sklearn.feature_extraction.text import TfidfTransformer
    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([('tfidf', TfidfTransformer()),
    ...                      ('chi2', SelectKBest(chi2, k=1000)),
    ...                      ('nb', MultinomialNB())])
    >>> classif = SklearnClassifier(pipeline)
"""

from nltk.classify.api import ClassifierI
from nltk.probability import DictionaryProbDist

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    pass

__all__ = ["SklearnClassifier"]


class SklearnClassifier(ClassifierI):
    """Wrapper for scikit-learn classifiers."""

    def __init__(self, estimator, dtype=float, sparse=True):
        """
        :param estimator: scikit-learn classifier object.

        :param dtype: data type used when building feature array.
            scikit-learn estimators work exclusively on numeric data. The
            default value should be fine for almost all situations.

        :param sparse: Whether to use sparse matrices internally.
            The estimator must support these; not all scikit-learn classifiers
            do (see their respective documentation and look for "sparse
            matrix"). The default value is True, since most NLP problems
            involve sparse feature sets. Setting this to False may take a
            great amount of memory.
        :type sparse: boolean.
        """
        self._clf = estimator
        self._encoder = LabelEncoder()
        self._vectorizer = DictVectorizer(dtype=dtype, sparse=sparse)


    def __repr__(self):
        return "<SklearnClassifier(%r)>" % self._clf

    def classify_many(self, featuresets):
        """Classify a batch of samples.

        :param featuresets: An iterable over featuresets, each a dict mapping
            strings to either numbers, booleans or strings.
        :return: The predicted class label for each input sample.
        :rtype: list
        """
        X = self._vectorizer.transform(featuresets)
        classes = self._encoder.classes_
        return [classes[i] for i in self._clf.predict(X)]


    def prob_classify_many(self, featuresets):
        """Compute per-class probabilities for a batch of samples.

        :param featuresets: An iterable over featuresets, each a dict mapping
            strings to either numbers, booleans or strings.
        :rtype: list of ``ProbDistI``
        """
        X = self._vectorizer.transform(featuresets)
        y_proba_list = self._clf.predict_proba(X)
        return [self._make_probdist(y_proba) for y_proba in y_proba_list]


    def labels(self):
        """The class labels used by this classifier.

        :rtype: list
        """
        return list(self._encoder.classes_)


    def train(self, labeled_featuresets):
        """
        Train (fit) the scikit-learn estimator.

        :param labeled_featuresets: A list of ``(featureset, label)``
            where each ``featureset`` is a dict mapping strings to either
            numbers, booleans or strings.
        """

        X, y = list(zip(*labeled_featuresets))
        X = self._vectorizer.fit_transform(X)
        y = self._encoder.fit_transform(y)
        self._clf.fit(X, y)

        return self


    def _make_probdist(self, y_proba):
        classes = self._encoder.classes_
        return DictionaryProbDist({classes[i]: p for i, p in enumerate(y_proba)})



if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB

    import nltk
    from nltk.classify.util import names_demo, names_demo_features

    pos_tweets = [('I love this this car car', 'positive'),
            ('This view is amazing', 'positive'),
            ('I feel great this morning', 'positive'),
            ('I am so excited about the concert', 'positive'),
            ('He is my best friend', 'positive')]

    neg_tweets = [('I do not like this car', 'negative'),
            ('This view is horrible', 'negative'),
            ('I feel tired this morning', 'negative'),
            ('I am not looking forward to the concert', 'negative'),
            ('He is my enemy', 'negative')]

    tweets = []
    for (words, sentiment) in pos_tweets + neg_tweets:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append((words_filtered, sentiment))


    def get_words_in_tweets(tweets):
        all_words = []
        for (words, sentiment) in tweets:
            all_words.extend(words)
        return all_words

    def get_word_features(wordlist):
        wordlist = nltk.FreqDist(wordlist)
        word_features = wordlist.keys()
        return word_features

    word_features = get_word_features(get_words_in_tweets(tweets))


    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
            #features['contains(%s)' % word] = document.count(word)
        return features

    training_set = nltk.classify.apply_features(extract_features, tweets)

    # print('training_set')
    # print(training_set)

    # classifier = nltk.NaiveBayesClassifier.train(training_set)
    classifier = SklearnClassifier(BernoulliNB(binarize=False))
    # classifier = SklearnClassifier(LogisticRegression(C=1000))

    classifier.train(training_set)

    test_tweets = [
            'I feel happy this morning', # positive
            'Larry is my good friend', # positive
            'I do not like that man', # negative
            'The house is not great', # negative
            'Your song is annoying'] # negative

    for test_tweet in test_tweets:
        print(test_tweet)
        # print(extract_features(test_tweet.split()))
        print(classifier.classify(extract_features(test_tweet.split())))
        print(classifier.prob_classify(extract_features(test_tweet.split())).prob('positive'))
        print(classifier.prob_classify(extract_features(test_tweet.split())).prob('negative'))
        print("=============================")


    # # Bernoulli Naive Bayes is designed for binary classification. We set the
    # # binarize option to False since we know we're passing boolean features.
    # print("scikit-learn Naive Bayes:")
    # names_demo(
            # SklearnClassifier(BernoulliNB(binarize=False)).train,
            # features=names_demo_features,
            # )

    # # The C parameter on logistic regression (MaxEnt) controls regularization.
    # # The higher it's set, the less regularized the classifier is.
    # print("\n\nscikit-learn logistic regression:")
    # names_demo(
            # SklearnClassifier(LogisticRegression(C=1000)).train,
            # features=names_demo_features,
            # )
