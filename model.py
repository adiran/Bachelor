import copy
import numpy as np

import functions as f


class Model(object):
    # Constructor

    _features = None
    _extractedFeatures = None
    _tolerance = None
    _name = None
    _score = -1.0
    _matches = -1
    _influencedBy = 0

    def __init__(self, features, tolerance, name, influencedBy):
        self.features = features
        self.extractedFeatures = copy.deepcopy(features)
        self.tolerance = tolerance
        self.name = name
        self.influencedBy = influencedBy


    @property
    def extractedFeatures(self):
        return self._extractedFeatures
   
    @property
    def features(self):
        return copy.deepcopy(self._features)

    @property
    def tolerance(self):
        return copy.deepcopy(self._tolerance)
    
    @property
    def name(self):
        return copy.deepcopy(self._name)
    
    @property
    def matches(self):
        return copy.deepcopy(self._matches)

    @property
    def influencedBy(self):
        return copy.deepcopy(self._influencedBy)

    @property
    def score(self):
        return copy.deepcopy(self._score)
    
    @features.setter
    def features(self, features):
        if self._features == None:
            self._features = features

    @extractedFeatures.setter
    def extractedFeatures(self, extractedFeatures):
        if self._extractedFeatures == None:
            self._extractedFeatures = []
            for i in extractedFeatures:
                self._extractedFeatures.append(f.extractFeatures(i))

    @tolerance.setter
    def tolerance(self, tolerance):
        if self._tolerance == None:
            self._tolerance = tolerance

    @name.setter
    def name(self, name):
        if self._name == None:
            self._name = name

    @matches.setter
    def matches(self, matches):
        if self._matches < 0:
            self._matches = matches

    @influencedBy.setter
    def influencedBy(self, influencedBy):
        self._influencedBy = influencedBy

    @score.setter
    def score(self, integer):
        if self._matches >= 0:
            self._score = float((float(self._matches) * float(self._influencedBy) * 10000000000) / float(self._tolerance))