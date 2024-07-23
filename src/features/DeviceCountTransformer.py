import sys
import pandas as pd
import logging

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class DeviceCountTransformer(BaseEstimator, TransformerMixin):
    """
    Pipeline feature transformer for device count computation
    Should be applied on columns  device_ip, device_id
    """

    def __init__(self, column_name: str):
        self.device_count_features = []
        self.column_name = column_name
        self.data_group = None

    def fit(self, X: pd.DataFrame, y=None):
        data_group = X[[self.column_name, "id"]].groupby([self.column_name]).count()

        for index in X[self.column_name]:
            self.device_count_features.append(data_group["id"][index])

        return self

    def transform(self, X: pd.DataFrame):
        return pd.DataFrame(
            self.device_count_features, columns=[f"{self.column_name}_count"]
        )