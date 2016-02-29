"""Audio Trainer v1.0"""
# import of own scripts
import copy
import os.path

class Model(object):
    # Constructor

    _features = None
    _threshold = None
    _name = None
    _score = -1.0
    _matches = -1
    _influencedBy = []
    _loaded = False
    _script = None

    def __init__(self, features, threshold, name, influencedBy, script):
        self.features = features
        self.extractedFeatures = copy.deepcopy(features)
        self.threshold = threshold
        self.name = name
        self.influencedBy = influencedBy
        self.script = script
   
    @property
    def features(self):
        return copy.deepcopy(self._features)

    @property
    def threshold(self):
        return copy.deepcopy(self._threshold)
    
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

    @property
    def script(self):
        if os.path.isfile(self._script):
            return copy.deepcopy(self._script)
        else:
            return None
    
    @features.setter
    def features(self, features):
        if self._features == None:
            self._features = features

    @threshold.setter
    def threshold(self, threshold):
        if self._threshold == None:
            self._threshold = threshold

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

    @script.setter
    def script(self, script):
        if os.path.isfile(script):
            self._script = script
        else:
            self._script = None

    def calculateScore(self):
        if self._matches >= 0:
            tmpThreshold = 0.
            tmpInfluenceCounter = 0.
            for i in self._threshold:
                tmpThreshold += i/len(self._threshold)
            for i in self._influencedBy:
                tmpInfluenceCounter += i/len(self._influencedBy)
            self._score = float((float(self._matches) * float(tmpInfluenceCounter)) / float(tmpThreshold))

    def activate(self):
        self._loaded = True

    def deactivate(self):
        self._loaded = False