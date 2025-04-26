import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.data_processing import DataProcessor # Assuming your DataProcessor is in src/data_processing.py
from sklearn.preprocessing import LabelEncoder # If needed for assertions

# Configure logging for tests (optional, but helps see logs during test runs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper: Create a dummy CSV file for testing ---
@pytest.fixture(scope="module")
def dummy_csv(tmp_path_factory):
    """Creates a temporary dummy CSV file for testing."""
    data = {
        'OBJECTID': [1, 2, 3, 4, 5, 6],
        'ORIGINL_RM': ['RM A', 'RM B', 'RM A', 'RM C', 'RM B', 'RM A'],
        'MANCON1': ['FW', 'W', 'T', 'FWT', np.nan, 'C'],
        'MANCON2': ['C', np.nan, 'B', 'F', 'W', 'B'],
        'MANCON3': [np.nan, np.nan, np.nan, 'T', np.nan, 'F'],
        'EXTENT1': [50, 100, 80, 70, 0, 60],
        'EXTENT2': [50, 0, 20, 30, 100, 40],
        'EXTENT3': [0, 0, 0, 0, 0, 0],
        'ERPOLY': ['N', 'L', 'M', 'S', np.nan, 'H'],
        'C_SLOPE': [22, 25, 6, 28, 16, -99], # Includes standard, special, missing codes
        'C_DRAIN': [21, 23, 24, 25, np.nan, 22],
        'C_AGRI': [21, 22, 23, 24, 25, 21], # Target column with some values
        'SHAPE_Length': [100.0, 200.0, 150.0, 120.0, 300.0, 180.0], # Example numerical column
        'Some_Other_Cat': ['A', 'B', 'A', 'C', 'B', 'A'] # Another categorical, no NaNs
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path_factory.mktemp("data") / "dummy_soil_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

# --- Tests for DataProcessor ---

def test_data_processor_initialization(dummy_csv: Path):
    """Test if the processor initializes and loads data."""
    processor = DataProcessor(data_path=dummy_csv, target_column='C_AGRI')
    assert processor.raw_data is not None
    assert not processor.raw_data.empty
    assert processor.raw_data.shape[0] == 6 # Check number of rows
    assert 'C_AGRI' in processor.raw_data.columns

def test_data_processor_initialization_missing_file():
    """Test if initialization raises error for missing file."""
    with pytest.raises(FileNotFoundError):
        DataProcessor(data_path=Path("non_existent_file.csv"), target_column='C_AGRI')

# Test specific preprocessing methods with small dataframes
# These tests check the output of individual transformation steps

def test_create_mancon_flags(dummy_csv: Path):
    """Test create_mancon_flags logic."""
    processor = DataProcessor(data_path=dummy_csv, target_column='C_AGRI')
    df = processor.raw_data.copy()
    processed_df = processor.create_mancon_flags(df)

    assert 'has_fine_texture' in processed_df.columns
    assert 'has_wetness' in processed_df.columns
    assert 'has_topography' in processed_df.columns
    assert 'has_coarse_texture' in processed_df.columns
    assert 'has_bedrock' in processed_df.columns
    assert 'has_no_constraints' in processed_df.columns # Check special cases
    assert 'has_rock' in processed_df.columns
    assert 'has_organic' in processed_df.columns # Added check for other flags
    assert 'has_marsh' in processed_df.columns
    assert 'has_water' in processed_df.columns

    # Corrected assertions based on dummy data and method logic
    # Row 1: FW -> has_fine_texture=1, has_wetness=1
    # Row 2: W -> has_wetness=1
    # Row 3: T -> has_topography=1
    # Row 4: FWT -> has_fine_texture=1, has_wetness=1, has_topography=1
    # Row 5: NaN, W -> has_wetness=1
    # Row 6: C, B, F -> has_coarse_texture=1, has_bedrock=1, has_fine_texture=1
    assert processed_df['has_fine_texture'].tolist() == [1, 0, 0, 1, 0, 1]
    assert processed_df['has_wetness'].tolist() == [1, 1, 0, 1, 1, 0]
    assert processed_df['has_topography'].tolist() == [0, 0, 1, 1, 0, 0]
    # Corrected assertion for has_coarse_texture - Row 0 has 'C' in MANCON2
    assert processed_df['has_coarse_texture'].tolist() == [1, 0, 0, 0, 0, 1]
    assert processed_df['has_bedrock'].tolist() == [0, 0, 1, 0, 0, 1]
    # Add more assertions based on expected flags for dummy data

def test_create_weighted_mancon(dummy_csv: Path):
    """Test create_weighted_mancon logic."""
    processor = DataProcessor(data_path=dummy_csv, target_column='C_AGRI')
    df = processor.raw_data.copy()
    processed_df = processor.create_weighted_mancon(df)

    assert 'W_MANCON1' in processed_df.columns
    assert 'W_MANCON2' in processed_df.columns
    assert 'W_MANCON3' in processed_df.columns
    assert 'Total_W_MANCON' in processed_df.columns

    # Mancon1: FW (count=2), Extent1: 50 -> 2 * 50 / 100 = 1.0
    # Mancon2: C (count=1), Extent2: 50 -> 1 * 50 / 100 = 0.5
    # Total for Row 1 = 1.0 + 0.5 + 0 = 1.5
    assert processed_df['W_MANCON1'].iloc[0] == pytest.approx(1.0)
    assert processed_df['W_MANCON2'].iloc[0] == pytest.approx(0.5)
    assert processed_df['W_MANCON3'].iloc[0] == pytest.approx(0.0)
    assert processed_df['Total_W_MANCON'].iloc[0] == pytest.approx(1.5)
    # Add more assertions for other rows/columns

def test_erpoly_processor(dummy_csv: Path):
    """Test erpoly_processor logic."""
    processor = DataProcessor(data_path=dummy_csv, target_column='C_AGRI')
    df = processor.raw_data.copy()
    processed_df = processor.erpoly_processor(df)

    assert 'encoded_ERPOLY' in processed_df.columns
    assert 'ERPOLY_missing' in processed_df.columns

    # N=1, L=2, M=3, S=5, nan=0, H=4
    assert processed_df['encoded_ERPOLY'].tolist() == [1, 2, 3, 5, 0, 4]
    assert processed_df['ERPOLY_missing'].tolist() == [0, 0, 0, 0, 1, 0]

def test_classification_processing(dummy_csv: Path):
    """Test classification_processing logic."""
    processor = DataProcessor(data_path=dummy_csv, target_column='C_AGRI')
    df = processor.raw_data.copy()

    processed_df = processor.classification_processing(df)

    assert 'encoded_C_SLOPE' in processed_df.columns
    assert 'C_SLOPE_is_water' in processed_df.columns
    assert 'C_SLOPE_is_urban' in processed_df.columns
    assert 'C_SLOPE_is_marsh' in processed_df.columns
    assert 'C_SLOPE_is_erodedSlope' in processed_df.columns

    assert 'encoded_C_DRAIN' in processed_df.columns
    assert 'C_DRAIN_is_water' in processed_df.columns
    assert 'C_DRAIN_is_urban' in processed_df.columns
    assert 'C_DRAIN_is_marsh' in processed_df.columns
    assert 'C_DRAIN_is_erodedSlope' in processed_df.columns

    # C_SLOPE: [22, 25, 6, 28, 16, -99]
    # encoded: [2, 5, 0, 8, 0, 0] (21-28 -> 1-8, others 0/NaN)
    assert processed_df['encoded_C_SLOPE'].tolist() == [2.0, 5.0, 0.0, 8.0, 0.0, 0.0] # Check encoded values
    # flags: [0, 0, 1, 0, 1, 0] (6 is water, 16 is urban)
    assert processed_df['C_SLOPE_is_water'].tolist() == [0, 0, 1, 0, 0, 0]
    assert processed_df['C_SLOPE_is_urban'].tolist() == [0, 0, 0, 0, 1, 0]
    assert processed_df['C_SLOPE_is_marsh'].tolist() == [0, 0, 0, 0, 0, 0] # No 13 in C_SLOPE dummy data
    assert processed_df['C_SLOPE_is_erodedSlope'].tolist() == [0, 0, 0, 0, 0, 0] # No 7 in C_SLOPE dummy data

    # C_DRAIN: [21, 23, 24, 25, np.nan, 22]
    # encoded: [1, 3, 4, 5, 0, 2] (21-28 -> 1-8, NaN -> 0)
    assert processed_df['encoded_C_DRAIN'].tolist() == [1.0, 3.0, 4.0, 5.0, 0.0, 2.0]
    # flags: [0, 0, 0, 0, 0, 0] (No special codes 6, 7, 13, 16 in C_DRAIN dummy data)
    assert processed_df['C_DRAIN_is_water'].tolist() == [0, 0, 0, 0, 0, 0]
    assert processed_df['C_DRAIN_is_urban'].tolist() == [0, 0, 0, 0, 0, 0]
    assert processed_df['C_DRAIN_is_marsh'].tolist() == [0, 0, 0, 0, 0, 0]
    assert processed_df['C_DRAIN_is_erodedSlope'].tolist() == [0, 0, 0, 0, 0, 0]

    # Add tests for other flags and columns like C_SALT, C_SURFTEXT if they were in dummy data
    # and are processed by classification_processing

# Test the overall preprocess method
def test_preprocess_overall(dummy_csv: Path):
    """Test the full preprocessing pipeline."""
    processor = DataProcessor(data_path=dummy_csv, target_column='C_AGRI')
    processed_df = processor.preprocess()

    assert processor.processed_data is not None
    assert not processed_df.empty

    # Check if expected final columns exist
    # Based on your 'keep' logic: ORIGINL_RM, Target, Newly Created, OHE
    # Need to list all expected output columns based on dummy data and processing
    expected_cols = {
        'ORIGINL_RM', 'C_AGRI', # Kept originals + target
        'has_fine_texture', 'has_wetness', 'has_topography', 'has_coarse_texture', # MANCON flags
        'has_bedrock', 'has_no_constraints', 'has_rock', 'has_organic', 'has_marsh', 'has_water',
        'W_MANCON1', 'W_MANCON2', 'W_MANCON3', 'Total_W_MANCON', # Weighted MANCON
        'encoded_ERPOLY', 'ERPOLY_missing', # ERPOLY
        'encoded_C_SLOPE', 'C_SLOPE_is_water', 'C_SLOPE_is_urban', 'C_SLOPE_is_marsh', 'C_SLOPE_is_erodedSlope', # C_SLOPE features
        'encoded_C_DRAIN', 'C_DRAIN_is_water', 'C_DRAIN_is_urban', 'C_DRAIN_is_marsh', 'C_DRAIN_is_erodedSlope', # C_DRAIN features
        # Removed C_AGRI flag columns as classification_processing does not process the target for flags
        'Some_Other_Cat_A', 'Some_Other_Cat_B', 'Some_Other_Cat_C', # Corrected expected OHE columns - no _Missing column
        # SHAPE_Length is numerical and not explicitly processed, should be dropped by the 'keep' logic
    }

    # Check if all expected columns are in the final DataFrame
    for col in expected_cols:
        assert col in processed_df.columns, f"Expected column '{col}' not found in processed data."

    # Check if columns *not* expected are dropped (e.g., original MANCONs, original C_SLOPE etc., SHAPE_Length)
    # Need to list original columns expected to be dropped
    original_cols_to_drop_examples = ['MANCON1', 'MANCON2', 'MANCON3', 'ERPOLY', 'C_SLOPE', 'C_DRAIN', 'SHAPE_Length', 'OBJECTID']
    for col in original_cols_to_drop_examples:
         # OBJECTID is explicitly excluded from dropping in preprocess if it's an object type and not in keep list.
         # Let's adjust this check based on the actual logic in preprocess.
         # The preprocess method drops columns NOT in cols_to_keep_final_set.
         # OBJECTID is not added to cols_to_keep_final_set unless it's the target.
         # So, OBJECTID should be dropped unless it's the target column.
         if col != 'OBJECTID' or col == processor.target_column:
             assert col not in processed_df.columns, f"Original column '{col}' should have been dropped but is present."
         # If OBJECTID is not the target and is an object type, it might be OHE'd if not excluded.
         # Your code explicitly excludes OBJECTID from OHE.
         # So, if OBJECTID is not the target and is not numeric, it should be dropped.
         if col == 'OBJECTID' and col != processor.target_column and not pd.api.types.is_numeric_dtype(processor.raw_data[col]):
              assert col not in processed_df.columns, f"Original column '{col}' should have been dropped but is present."


    # Check target encoding result (C_AGRI: [21, 22, 23, 24, 25, 21]) -> sorted unique [21, 22, 23, 24, 25] -> [0, 1, 2, 3, 4]
    # Assuming C_AGRI values are already numeric and 21-25 as per dummy data
    # If C_AGRI original values are strings like 'Class 1', 'Class 2', the encoding will be different
    # Based on the sample C_AGRI [21, 22, 23, 24, 25, 21], the encoded values should be [0, 1, 2, 3, 4, 0]
    # Check if the target column is now integer type
    assert pd.api.types.is_integer_dtype(processed_df['C_AGRI'])
    assert processed_df['C_AGRI'].tolist() == [0, 1, 2, 3, 4, 0]
    # Check the label encoder mapping - Expecting strings because the preprocess method converts to string before fitting
    assert list(processor.label_encoder.classes_) == ['21', '22', '23', '24', '25']


# Test the split_data_three_way method
def test_split_data_three_way(dummy_csv: Path):
    """Test the three-way split logic."""
    processor = DataProcessor(data_path=dummy_csv, target_column='C_AGRI', random_state=42)
    processed_df = processor.preprocess() # Ensure data is preprocessed

    test_size = 0.5 # Use a larger test size for a tiny dummy dataset to make split noticeable
    # The split_data_three_way method now checks for minimum class size before stratifying,
    # so we don't need to pass stratify=None here.
    X_train_val, X_test, y_train_val, y_test = processor.split_data_three_way(test_size=test_size, random_state=42)

    # Check shapes
    total_rows = processed_df.shape[0]
    expected_test_rows = int(total_rows * test_size)
    expected_train_val_rows = total_rows - expected_test_rows

    assert X_train_val.shape[0] == expected_train_val_rows
    assert y_train_val.shape[0] == expected_train_val_rows
    assert X_test.shape[0] == expected_test_rows
    assert y_test.shape[0] == expected_test_rows

    # Check if feature columns match processed_df columns minus target
    expected_feature_cols = set(processed_df.columns) - {'C_AGRI'}
    assert set(X_train_val.columns) == expected_feature_cols
    assert set(X_test.columns) == expected_feature_cols

    # Check for overlap (should be none between train+val and test)
    # Use index to check for row overlap
    assert len(set(X_train_val.index).intersection(set(X_test.index))) == 0

    # Stratification was skipped in the method due to small class size,
    # so we don't assert stratification proportions here for this dummy data.
    # For real data or larger dummy data, you would add assertions for value_counts proportions.
