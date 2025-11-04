"""Example of using the .fit() method directly with mocked data."""
import pandas as pd
import numpy as np
from multitabfm.core import MultiTabFM

def main():
    """Example of using the .fit() method directly with mocked data."""
    
    # 1. Create mock training data and test data
    # In a real scenario, this would be your feature-engineered training data
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

    label_column = 'target'
    task_type = 'regression'

    # 2. Instantiate MultiTabFM and fit the model
    # For this example, we don't need DFS since we are providing the features directly.
    model_config = {
        "hyperparameters": {
            "TABPFNV2": {
                "n_estimators": 2,
            }
        },
    }
    engine = MultiTabFM(model_config=model_config)
    
    print("Fitting model with mocked data...")
    engine.fit(train_data, label_column=label_column, task_type=task_type)
    print("Model fitting complete.")

    # 3. Create mock test data and predict
    # The test data should have the same feature columns as the training data (excluding the label)
    print("\nMaking predictions on mocked test data...")
    predictions = engine.predict(test_features)
    print("Predictions made:")
    print(predictions.head())


if __name__ == "__main__":
    main()
