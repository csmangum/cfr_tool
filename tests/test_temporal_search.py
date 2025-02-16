import unittest
from scripts.search_regulations import TemporalSearchEnhancer

class TestTemporalSearchEnhancer(unittest.TestCase):

    def setUp(self):
        self.enhancer = TemporalSearchEnhancer()

    def test_extract_temporal_constraints_iso_date(self):
        query = "regulations from 2020-01-01 to 2023-12-31"
        expected = {
            "date_range": {"start": "2020-01-01", "end": "2023-12-31"},
            "version_type": "current|historical|all",
            "temporal_context": "before|after|during"
        }
        result = self.enhancer.extract_temporal_constraints(query)
        self.assertEqual(result["date_range"], expected["date_range"])

    def test_extract_temporal_constraints_keywords(self):
        query = "regulations before 2020"
        expected = {
            "date_range": {"start": None, "end": "2020-01-01"},
            "version_type": "current|historical|all",
            "temporal_context": "before|after|during"
        }
        result = self.enhancer.extract_temporal_constraints(query)
        self.assertEqual(result["temporal_context"], expected["temporal_context"])

    def test_extract_temporal_constraints_version(self):
        query = "current regulations"
        expected = {
            "date_range": {"start": None, "end": None},
            "version_type": "current",
            "temporal_context": "before|after|during"
        }
        result = self.enhancer.extract_temporal_constraints(query)
        self.assertEqual(result["version_type"], expected["version_type"])

if __name__ == "__main__":
    unittest.main()
