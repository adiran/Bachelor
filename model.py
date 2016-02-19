import copy
import numpy as np

import functions as f


class Model(object):
    # Constructor

    _features = None
    _tolerance = None
    _name = None
    _score = -1.0
    _matches = -1
    _influencedBy = 0
    _loaded = False

    def __init__(self, features, tolerance, name, influencedBy):
        self.features = features
        self.extractedFeatures = copy.deepcopy(features)
        self.tolerance = tolerance
        self.name = name
        self.influencedBy = influencedBy
   
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

    @property
    def loaded(self):
        return copy.deepcopy(self._loaded)
    
    @features.setter
    def features(self, features):
        if self._features == None:
            self._features = features

    @tolerance.setter
    def tolerance(self, tolerance):
        if self._tolerance == None:
            self._tolerance = tolerance

    @name.setter
    def name(self, name):
        self._name = name

    @matches.setter
    def matches(self, matches):
        if self._matches < 0:
            self._matches = matches

    @influencedBy.setter
    def influencedBy(self, influencedBy):
        self._influencedBy = influencedBy

    def calculateScore(self):
        if self._matches >= 0:
            self._score = float((float(self._matches) * float(self._influencedBy)) / float(self._tolerance))

    def activate(self):
        self._loaded = True

    def deactivate(self):
        self._loaded = False