"""Example of using the .fit() method directly with mocked data."""
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier,TabPFNRegressor
from LimiX.inference.predictor import LimiXPredictor
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

# set customized model
class CustomTabPFN(AbstractModel):
    def __init__(self, **kwargs):
        # Simply pass along kwargs to parent, and init our internal `_feature_generator` variable to None
        super().__init__(**kwargs)
        self._feature_generator = None

    # The `_preprocess` method takes the input data and transforms it to the internal representation usable by the model.
    # `_preprocess` is called by `preprocess` and is used during model fit and model inference.
    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            # This converts categorical features to numeric via stateful label encoding.
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self,
             X: pd.DataFrame,  # training data
             y: pd.Series,  # training labels
             **kwargs):  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
        print('Entering the `_fit` method')
        X = self.preprocess(X, is_train=True)
        self.model = TabPFNRegressor(model_path="/root/autodl-tmp/tabpfn_2_5/tabpfn-v2.5-regressor-v2.5_default.ckpt")  # Uses TabPFN 2.5 weights, finetuned on real data.
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

class LimixPFN(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            # This converts categorical features to numeric via stateful label encoding.
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self,
             X: pd.DataFrame,  # training data
             y: pd.Series,  # training labels
             **kwargs):  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
        print('Entering the `_fit` method')
        X = self.preprocess(X, is_train=True)
        model_path = hf_hub_download(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", local_dir="./cache")
        self.model = LimiXPredictor(device='cuda', model_path=model_path) 
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

# load train and test data
train_data = pd.DataFrame({
    'user_id': [1, 2, 1, 3, 2, 4, 1, 2, 3, 4],
    'item_id': [101, 102, 103, 101, 104, 105, 106, 107, 108, 109],
    'feature_1': np.random.rand(10),
    'feature_2': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
    'target': np.random.rand(10) * 100
})

test_features = pd.DataFrame({
    'user_id': [1, 4],
    'item_id': [106, 102],
    'feature_1': np.random.rand(2),
    'feature_2': ['B', 'C'],
})

label = 'target'
task_type = 'regression'

# clean labels
X = train_data.drop(columns=[label])
y = train_data[label]

from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
# Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
problem_type = infer_problem_type(y=y)  # Infer problem type (or else specify directly)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_clean = label_cleaner.transform(y)

print(f'Labels cleaned: {label_cleaner.inv_map}')
print(f'inferred problem type as: {problem_type}')
print('Cleaned label values:')
y_clean.head(5)

# clean features
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
set_logger_verbosity(2)  # Set logger so more detailed logging is shown for tutorial

feature_generator = AutoMLPipelineFeatureGenerator()
X_clean = feature_generator.fit_transform(X)

X_clean.head(5)

# fit model
custom_model = CustomTabPFN()
custom_model.fit(X=X_clean, y=y_clean)  # Fit custom model

# test prediction
X_test_clean = feature_generator.transform(test_features)
predictions = custom_model.predict(X_test_clean)
print("Predictions made:")
print(predictions)  