import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from etl_scripts import extract_data, transform_data, load_data

class TestETLScripts(unittest.TestCase):
    def setUp(self):
        # Sample data setup for use in tests
        self.sample_data = pd.DataFrame({
            'text_column': ['TEST', 'Data', '', '123']
        })
        self.transformed_data = pd.DataFrame({
            'text_column': ['test', 'data', '', '123']
        })

    @patch('pandas.read_csv')
    def test_extract_data(self, mock_read_csv):
        # Setup mock to return a DataFrame
        mock_read_csv.return_value = self.sample_data
        result = extract_data('fake/path.csv')
        mock_read_csv.assert_called_with('fake/path.csv')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_data))

    def test_transform_data(self):
        # Test transformation logic
        result = transform_data(self.sample_data)
        pd.testing.assert_frame_equal(result, self.transformed_data)

    @patch('pandas.DataFrame.to_csv')
    def test_load_data(self, mock_to_csv):
        # Test loading data with mocking to prevent actual file writing
        output_path = 'fake/output_path.csv'
        load_data(self.transformed_data, output_path)
        mock_to_csv.assert_called_with(output_path, index=False)

    def test_error_handling(self):
        # Test error handling for an empty DataFrame
        empty_df = pd.DataFrame()
        transformed_empty = transform_data(empty_df)
        self.assertTrue(transformed_empty.empty)

if __name__ == '__main__':
    unittest.main()