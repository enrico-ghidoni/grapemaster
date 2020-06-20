import numpy as np
import pandas as pd


class AdjectiveColumnIndicators(object):
    def _set_reference_df(self, df):
        """Set reference DataFrame with the same columns as the classificator, only used for setup"""
        self._reference_df = df

    def preprocess(self, data):
        """Return an array with values 1 for corresponding adjectives passed in `data` and 0 in other indicator columns"""
        df_cols = [f'adj_{x}' for x in data]

        cr_df = self._reference_df.copy(deep=True)
        cr_df[df_cols] = 1

        return cr_df.values
