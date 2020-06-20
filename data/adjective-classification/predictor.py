import os
import pickle

from sklearn.externals import joblib


class AdjectiveVarietyPredictor(object):
    def __init__(self, model, preprocessor):
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        preprocessed_input = self._preprocessor.preprocess(instances)

        return list(self._model.predict(preprocessed_input))

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'model.joblib')
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')

        model = joblib.load(model_path)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        return cls(model, preprocessor)
