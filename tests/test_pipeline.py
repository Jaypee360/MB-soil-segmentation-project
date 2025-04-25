import pytest
import logging
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock # For mocking
from src.pipeline import run_pipeline_three_way_split # Import the function

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper: Create a dummy data file ---
@pytest.fixture(scope="module")
def dummy_data_file(tmp_path_factory):
    """Creates a temporary dummy CSV file for testing."""
    data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'C_AGRI': [0, 1, 0]} # Simple dummy data
    df = pd.DataFrame(data)
    csv_path = tmp_path_factory.mktemp("data") / "pipeline_dummy_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# --- Tests for pipeline.py ---

@patch('pipeline.DataProcessor') # Mock the DataProcessor class
@patch('pipeline.XGBoostTrainer') # Mock the XGBoostTrainer class
def test_run_pipeline_success(mock_xgb_trainer_class, mock_data_processor_class, dummy_data_file: Path):
    """Tests successful execution of the pipeline."""

    # Configure the mocked DataProcessor instance and its methods
    mock_processor_instance = MagicMock()
    mock_data_processor_class.return_value = mock_processor_instance # __init__ returns this mock instance

    # Configure the return value of the preprocess method
    mock_processed_data = pd.DataFrame({'feature1': [1,2], 'target': [0,1]}) # Dummy processed data
    mock_processor_instance.preprocess.return_value = mock_processed_data

    # Configure the return value of the split_data_three_way method
    mock_X_train_val = pd.DataFrame({'f1': [1]})
    mock_X_test = pd.DataFrame({'f1': [2]})
    mock_y_train_val = pd.Series([0])
    mock_y_test = pd.Series([1])
    mock_processor_instance.split_data_three_way.return_value = (
        mock_X_train_val, mock_X_test, mock_y_train_val, mock_y_test
    )

    # Configure the mocked XGBoostTrainer instance and its methods
    mock_trainer_instance = MagicMock()
    mock_xgb_trainer_class.return_value = mock_trainer_instance # __init__ returns this mock instance

    # Configure the return value of the optimize_hyperparameters method
    mock_optimization_results = {'best_params': {'eta': 0.1}}
    mock_trainer_instance.optimize_hyperparameters.return_value = mock_optimization_results

    # Configure the return value of the train method
    mock_trained_model = MagicMock()
    mock_trainer_instance.train.return_value = mock_trained_model

    # Configure the return value of the evaluate method
    mock_test_metrics = {'final_test_accuracy': 0.99}
    mock_trainer_instance.evaluate.return_value = mock_test_metrics

    # Run the pipeline function
    run_pipeline_three_way_split(data_filepath=str(dummy_data_file), target_column='C_AGRI')

    # --- Assertions: Check if methods were called in the correct order ---

    # DataProcessor should be initialized
    mock_data_processor_class.assert_called_once_with(data_path=Path(dummy_data_file), target_column='C_AGRI')

    # preprocess should be called
    mock_processor_instance.preprocess.assert_called_once()

    # split_data_three_way should be called after preprocess
    mock_processor_instance.split_data_three_way.assert_called_once_with(test_size=0.2) # Check default test_size

    # XGBoostTrainer should be initialized
    mock_xgb_trainer_class.assert_called_once_with(experiment_name="soil_viability_optimization_cv")

    # optimize_hyperparameters should be called with train_val data
    # Need to verify it was called with the data returned by split_data_three_way
    mock_trainer_instance.optimize_hyperparameters.assert_called_once_with(
        X=mock_X_train_val,
        y=mock_y_train_val,
        n_trials=50, # Check default
        cv_folds=5   # Check default
    )

    # A new XGBoostTrainer instance for final model (as per current pipeline code)
    # This might need adjustment depending on whether you re-instantiate or reuse
    # Current code creates final_model_trainer = XGBoostTrainer(...)
    final_mock_trainer_instance = MagicMock()
    # If XGBoostTrainer is called twice, check the second call
    calls = mock_xgb_trainer_class.call_args_list
    assert len(calls) == 2
    mock_xgb_trainer_class.assert_any_call(experiment_name="soil_viability_final_model")
    # Update mock trainer instance to the one used for final model if different
    if calls[0] != calls[1]: # Check if it was called twice with different args
        # Find the instance created with the 'final_model' experiment name
         for instance_call in mock_xgb_trainer_class.mock_calls:
             if 'soil_viability_final_model' in instance_call[1]:
                 final_mock_trainer_instance = mock_xgb_trainer_class.return_value # Assuming it's always the same mock object
                 break # This part of mocking call order can be tricky

    # train should be called with train_val data and best params
    expected_final_params = mock_optimization_results['best_params']
    # If you use default params as fallback, you might need to check those too
    final_mock_trainer_instance.train.assert_called_once_with(mock_X_train_val, mock_y_train_val, params=expected_final_params)


    # evaluate should be called with the trained model and test data
    final_mock_trainer_instance.evaluate.assert_called_once_with(mock_trained_model, mock_X_test, mock_y_test, dataset_name='final_test')

    # You can add assertions to check logging output if needed, but that's more advanced mocking

# Add tests for error cases (e.g., DataProcessor.preprocess raises error, XGBoostTrainer.train raises error)
@patch('pipeline.DataProcessor')
@patch('pipeline.XGBoostTrainer')
def test_run_pipeline_preprocessing_fails(mock_xgb_trainer_class, mock_data_processor_class, dummy_data_file: Path):
    """Tests pipeline handles error during preprocessing."""
    mock_processor_instance = MagicMock()
    mock_data_processor_class.return_value = mock_processor_instance
    mock_processor_instance.preprocess.side_effect = RuntimeError("Preprocessing failed") # Make preprocess raise an error

    # Ensure the pipeline function exits gracefully (or re-raises the specific error if preferred)
    with pytest.raises(RuntimeError, match="Preprocessing failed"): # Expecting the error to be re-raised
         run_pipeline_three_way_split(data_filepath=str(dummy_data_file), target_column='C_AGRI')

    # Check that subsequent steps were NOT called
    mock_processor_instance.split_data_three_way.assert_not_called()
    mock_xgb_trainer_class.assert_not_called()


# Add more tests for other failure points (splitting fails, training fails, etc.)