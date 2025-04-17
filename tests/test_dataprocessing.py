import pytest
import pandas as pd
from pathlib import Path
from src.data_processing import DataProcessor  


@pytest.fixture
def csv_files(tmp_path):
    """Creates temporary CSV files for testing."""
    valid_file = tmp_path / "valid_data.csv"
    malformed_file = tmp_path / "malformed_data.csv"
    pd.DataFrame({
        "SOIL_CODE1": ["A1", "B2"],
        "SOILNAME1": ["Soil A", "Soil B"],
        "C_SLOPE": [22, 24],
        "C_DRAIN": [21, 23],
        "AGRI_CAP1": [1, 2],
    }).to_csv(valid_file, index=False)

    with open(malformed_file, "w") as f:
        f.write("SOIL_CODE1,SOILNAME1,C_SLOPE\n")
        f.write("A1,Soil A,22\nB2,Soil B,not_a_number\n")

    return valid_file, malformed_file


def test_load_data(csv_files):
    """Test loading data with valid and malformed files."""
    valid_file, malformed_file = csv_files

    # Test valid file
    processor = DataProcessor(data_path=valid_file, target_column="AGRI_CAP1")
    df = processor.load_data()
    assert df.shape == (2, 5), "Valid CSV should load correctly"

    # Test invalid file path
    with pytest.raises(Exception, match="Error loading data"):
        DataProcessor(data_path=Path("invalid_file.csv"), target_column="AGRI_CAP1").load_data()

    # Test malformed file
    with pytest.raises(Exception, match="Error loading data"):
        DataProcessor(data_path=malformed_file, target_column="AGRI_CAP1").load_data()

@pytest.fixture
def sample_data():
    """Fixture for creating sample data for MANCON flag tests."""
    return pd.DataFrame({
        "MANCON1": ["F", "W T", "NO CONSTRAINTS"],
        "MANCON2": ["C", "B", "ROCK"],
        "MANCON3": [None, "ORGANIC", "MARSH"],
    })


def test_create_mancon_flags(sample_data):
    """Test the creation of binary flags for management considerations."""
    processor = DataProcessor(data_path=None)  # No file path needed for this test
    result = processor.create_mancon_flags(sample_data)

    # Check if the flags are created correctly
    expected_flags = [
        "has_fine_texture",
        "has_wetness",
        "has_topography",
        "has_coarse_texture",
        "has_bedrock",
        "has_no_constraints",
        "has_rock",
        "has_organic",
        "has_marsh",
        "has_water",
    ]
    assert all(flag in result.columns for flag in expected_flags), "Not all expected flags were created"

    # Validate individual flag values
    assert result.loc[0, "has_fine_texture"] == 1, "Row 0 should have 'has_fine_texture' flag set"
    assert result.loc[1, "has_wetness"] == 1, "Row 1 should have 'has_wetness' flag set"
    assert result.loc[2, "has_no_constraints"] == 1, "Row 2 should have 'has_no_constraints' flag set"
    assert result.loc[2, "has_rock"] == 1, "Row 2 should have 'has_rock' flag set"
    assert result.loc[1, "has_organic"] == 1, "Row 1 should have 'has_organic' flag set"


@pytest.fixture
def sample_weighted_data():
    """Fixture for creating sample data for weighted MANCON tests."""
    return pd.DataFrame({
        "MANCON1": ["FWT", "C", "B"],
        "MANCON2": ["", "W", "T"],
        "MANCON3": [None, "C", "F"],
        "EXTENT1": [50, 30, 20],
        "EXTENT2": [30, 40, 30],
        "EXTENT3": [20, 30, 50],
    })


def test_create_weighted_mancon(sample_weighted_data):
    """Test the creation of weighted management considerations."""
    processor = DataProcessor(data_path=None)  # No file path needed for this test
    result = processor.create_weighted_mancon(sample_weighted_data)

    # Check if the Total_W_MANCON column is created
    assert "Total_W_MANCON" in result.columns, "The 'Total_W_MANCON' column was not created"

    # Validate calculated weighted scores for Total_W_MANCON
    expected_scores = [
        (3 * 50 / 100) + (0 * 30 / 100) + (0 * 20 / 100),  # Row 0: FWT
        (1 * 30 / 100) + (1 * 40 / 100) + (1 * 30 / 100),  # Row 1: C, W, C
        (1 * 20 / 100) + (1 * 30 / 100) + (1 * 50 / 100),  # Row 2: B, T, F
    ]

    assert result["Total_W_MANCON"].tolist() == pytest.approx(expected_scores), \
        "The weighted management scores are not correctly calculated"

    # Validate individual W_MANCON columns
    for i, (mancon, extent) in enumerate(zip(["MANCON1", "MANCON2", "MANCON3"], ["EXTENT1", "EXTENT2", "EXTENT3"])):
        expected_w_col = f"W_{mancon}"
        assert expected_w_col in result.columns, f"The column '{expected_w_col}' was not created"
        assert result[expected_w_col].notna().all(), f"Missing values found in '{expected_w_col}'"


@pytest.fixture
def erpoly_sample_data():
    """Fixture for creating sample data for ERPOLY processing."""
    return pd.DataFrame({
        "ERPOLY": ["N", "L", "M", "H", "S", None, "INVALID"]
    })


def test_erpoly_processor(erpoly_sample_data):
    """Test the ERPOLY processing method."""
    processor = DataProcessor(data_path=None)  # No file path needed for this test
    result = processor.erpoly_processor(erpoly_sample_data)

    # Check if the encoded_ERPOLY column is created
    assert "encoded_ERPOLY" in result.columns, "The 'encoded_ERPOLY' column was not created"

    # Validate encoded values
    expected_encoded = [1, 2, 3, 4, 5, 0, 0]
    assert result["encoded_ERPOLY"].tolist() == expected_encoded, "Encoded ERPOLY values are incorrect"

    # Check if the ERPOLY_missing flag column is created
    assert "ERPOLY_missing" in result.columns, "The 'ERPOLY_missing' column was not created"

    # Validate missing flags
    expected_missing = [0, 0, 0, 0, 0, 1, 0]
    assert result["ERPOLY_missing"].tolist() == expected_missing, "ERPOLY_missing flags are incorrect"


@pytest.fixture
def classification_sample_data():
    """Fixture for creating sample data for classification processing."""
    return pd.DataFrame({
        "C_SLOPE": [21, 6, 13, 28, 16],
        "C_DRAIN": [22, 7, 25, 20, None],
        "C_SALT": [23, 16, 24, 6, 13],
        "C_SURFTEXT": [26, 13, 7, 21, 28]
    })


def test_classification_processing(classification_sample_data):
    """Test the classification processing method."""
    processor = DataProcessor(data_path=None)  # No file path needed for this test
    result = processor.classification_processing(classification_sample_data)

    # Check if encoded columns are created
    encoded_columns = [
        "encoded_C_SLOPE", "encoded_C_DRAIN",
        "encoded_C_SALT", "encoded_C_SURFTEXT"
    ]
    for col in encoded_columns:
        assert col in result.columns, f"Encoded column '{col}' was not created"

    # Validate encoded values
    expected_encoded_values = {
        "encoded_C_SLOPE": [1, 0, 0, 8, 0],
        "encoded_C_DRAIN": [2, 0, 5, 0, 0],
        "encoded_C_SALT": [3, 0, 4, 0, 0],
        "encoded_C_SURFTEXT": [6, 0, 0, 1, 8]
    }
    for col, expected_values in expected_encoded_values.items():
        assert result[col].tolist() == expected_values, f"Values in '{col}' are incorrect"

    # Check if binary flags are created and validate their values
    flag_columns = {
        "C_SLOPE_is_water": [0, 0, 0, 0, 0],
        "C_SLOPE_is_urban": [0, 0, 0, 0, 1],
        "C_SLOPE_is_marsh": [0, 0, 1, 0, 0],
        "C_SLOPE_is_erodedSlope": [0, 1, 0, 0, 0],
    }
    for col, expected_values in flag_columns.items():
        assert col in result.columns, f"Flag column '{col}' was not created"
        assert result[col].tolist() == expected_values, f"Values in '{col}' are incorrect"
