import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock # For mocking
import xgboost as xgb
from src.train_evaluate import XGBoostTrainer # Assuming your class is here

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper: Create dummy data splits ---
@pytest.fixture(scope="module")
def dummy_data_splits():
    """Creates dummy data splits for testing."""
    X_train = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y_train = pd.Series(np.random.randint(0, 3, 100)) # 3 classes
    X_test = pd.DataFrame(np.random.rand(50, 10), columns=[f'feature_{i}' for i in range(10)])
    y_test = pd.Series(np.random.randint(0, 3, 50))
    return X_train, X_test, y_train, y_test

# --- Tests for XGBoostTrainer ---

# Patch mlflow methods globally for tests in this file
@patch('mlflow.set_tracking_uri')
@patch('mlflow.xgboost.autolog')
@patch('mlflow.start_run')
@patch('mlflow.log_params')
@patch('mlflow.log_metrics')
@patch('mlflow.xgboost.log_model')
@patch('mlflow.register_model')
@patch('mlflow.log_artifact')
@patch('matplotlib.pyplot.savefig') # Mock plotting functions
@patch('matplotlib.pyplot.close')
@patch('seaborn.heatmap')
@patch('seaborn.barplot')
@patch('pandas.DataFrame.to_csv') # Mock saving history dataframe in optimize_hyperparameters
def test_trainer_initialization(
    mock_to_csv, mock_barplot, mock_heatmap, mock_plt_close, mock_plt_savefig,
    mock_log_artifact, mock_register_model, mock_log_model, mock_log_metrics, mock_log_params, mock_start_run,
    mock_autolog, mock_set_tracking_uri
):
    """Test if the trainer initializes and sets up MLflow."""
    trainer = XGBoostTrainer(experiment_name="test_exp")
    mock_set_tracking_uri.assert_called_once()
    mock_autolog.assert_called_once()
    assert trainer.experiment_name == "test_exp"


@patch('mlflow.start_run')
@patch('mlflow.log_params')
@patch('mlflow.xgboost.log_model')
def test_trainer_train(
    mock_log_model, mock_log_params, mock_start_run,
    dummy_data_splits # Use the fixture
):
    """Test the model training method."""
    X_train, X_test, y_train, y_test = dummy_data_splits
    trainer = XGBoostTrainer(experiment_name="test_exp")

    # Mock the start_run context manager
    mock_start_run.return_value.__enter__.return_value = MagicMock(info=MagicMock(run_id="dummy_run_id"))

    # Mock the actual xgb.XGBClassifier fit method
    with patch('xgboost.XGBClassifier') as mock_xgb_classifier:
        mock_model_instance = MagicMock()
        mock_xgb_classifier.return_value = mock_model_instance

        params = {'eta': 0.1, 'max_depth': 3}
        trained_model = trainer.train(X_train, y_train, params=params)

        # Check if XGBClassifier was instantiated with expected params (and num_class added)
        expected_params = params.copy()
        expected_params['num_class'] = y_train.nunique() # Check if num_class is set
        mock_xgb_classifier.assert_called_once_with(**expected_params, random_state=42)

        # Check if fit was called with correct data and eval_set
        mock_model_instance.fit.assert_called_once()
        args, kwargs = mock_model_instance.fit.call_args
        assert args[0] is X_train
        assert args[1] is y_train
        assert 'eval_set' in kwargs
        assert kwargs['early_stopping_rounds'] == 10

        # Check if MLflow methods were called
        mock_start_run.assert_called_once_with(nested=True)
        mock_log_params.assert_called_once_with(expected_params)
        mock_log_model.assert_called_once_with(mock_model_instance, "soil_capability_classifier")

        # Check return value
        assert trained_model is mock_model_instance

# Add tests for train method error handling (e.g., mismatched shapes)
def test_trainer_train_mismatched_shapes():
    """Test train method error handling for mismatched data shapes."""
    trainer = XGBoostTrainer(experiment_name="test_exp")
    X_train = pd.DataFrame(np.random.rand(100, 10))
    y_train = pd.Series(np.random.randint(0, 3, 99)) # Mismatched
    with pytest.raises(ValueError, match="Feature and target sizes don't match"):
        trainer.train(X_train, y_train)


@patch('mlflow.start_run')
@patch('mlflow.log_metrics')
@patch('mlflow.log_artifact') # Mock log_artifact for plots/feature importance
@patch('xgboost.XGBClassifier.predict') # Mock the predict call
@patch('sklearn.metrics.accuracy_score') # Mock metric calculations
@patch('sklearn.metrics.f1_score')
@patch('sklearn.metrics.confusion_matrix')
@patch('matplotlib.pyplot.savefig') # Mock plotting
@patch('matplotlib.pyplot.close')
@patch('seaborn.heatmap')
@patch('seaborn.barplot')
def test_trainer_evaluate(
    mock_barplot, mock_heatmap, mock_plt_close, mock_plt_savefig, mock_confusion_matrix, mock_f1_score, mock_accuracy_score,
    mock_log_artifact, mock_log_metrics, mock_start_run, mock_predict,
    dummy_data_splits # Use fixture
):
    """Test the model evaluation method."""
    X_train, X_test, y_train, y_test = dummy_data_splits
    trainer = XGBoostTrainer(experiment_name="test_exp")
    mock_model = MagicMock() # Mock the trained model object
    mock_model.feature_importances_ = np.array([0.1] * 10) # Add dummy importances
    mock_model.predict.return_value = y_test # Mock predictions to be perfect for simplicity

    # Mock MLflow run context
    mock_start_run.return_value.__enter__.return_value = MagicMock()

    # Mock metric return values
    mock_accuracy_score.return_value = 0.95
    mock_f1_score.side_effect = [0.94, 0.93] # weighted, then macro

    metrics = trainer.evaluate(mock_model, X_test, y_test, dataset_name='test')

    # Check if predict was called
    mock_model.predict.assert_called_once_with(X_test)

    # Check if metrics were calculated
    mock_accuracy_score.assert_called_once_with(y_test, y_test)
    mock_f1_score.assert_any_call(y_test, y_test, average='weighted')
    mock_f1_score.assert_any_call(y_test, y_test, average='macro')
    mock_confusion_matrix.assert_called_once_with(y_test, y_test)

    # Check if MLflow metrics were logged
    mock_log_metrics.assert_called_once_with({
        'test_accuracy': 0.95,
        'test_weighted_f1': 0.94,
        'test_macro_f1': 0.93
    })

    # Check if plots and artifacts were logged
    mock_plt_savefig.assert_any_call('confusion_matrix_test.png')
    mock_log_artifact.assert_any_call('confusion_matrix_test.png')
    mock_plt_savefig.assert_any_call('feature_importance.png')
    mock_log_artifact.assert_any_call('feature_importance.png')

    # Check the returned metrics
    assert metrics == {
        'test_accuracy': 0.95,
        'test_weighted_f1': 0.94,
        'test_macro_f1': 0.93
    }

# Add tests for optimize_hyperparameters and cross_validate similarly,
# mocking optuna, sklearn.model_selection methods, and mlflow logging.