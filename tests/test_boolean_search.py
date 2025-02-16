import unittest
from scripts.search_regulations import BooleanQueryProcessor

class TestBooleanQueryProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = BooleanQueryProcessor()

    def test_parse_query_simple(self):
        query = "FOIA AND request"
        expected = {
            "operators": ["AND"],
            "terms": ["FOIA", "request"],
            "groups": []
        }
        result = self.processor.parse_query(query)
        self.assertEqual(result["operators"], expected["operators"])
        self.assertEqual(result["terms"], expected["terms"])
        self.assertEqual(result["groups"], expected["groups"])

    def test_parse_query_complex(self):
        query = "FOIA AND (request OR application) NOT expedited"
        expected = {
            "operators": ["AND", "OR", "NOT"],
            "terms": ["FOIA", "request", "application", "expedited"],
            "groups": ["(request OR application)"]
        }
        result = self.processor.parse_query(query)
        self.assertEqual(result["operators"], expected["operators"])
        self.assertEqual(result["terms"], expected["terms"])
        self.assertEqual(result["groups"], expected["groups"])

    def test_combine_results_and(self):
        results_sets = [
            [("doc1", 0.9, {}), ("doc2", 0.8, {})],
            [("doc2", 0.85, {}), ("doc3", 0.7, {})]
        ]
        operator = "AND"
        expected = [("doc2", 0.825, {})]
        result = self.processor.combine_results(results_sets, operator)
        self.assertEqual(result, expected)

    def test_combine_results_or(self):
        results_sets = [
            [("doc1", 0.9, {}), ("doc2", 0.8, {})],
            [("doc2", 0.85, {}), ("doc3", 0.7, {})]
        ]
        operator = "OR"
        expected = [("doc1", 0.9, {}), ("doc2", 0.85, {}), ("doc3", 0.7, {})]
        result = self.processor.combine_results(results_sets, operator)
        self.assertEqual(result, expected)

    def test_combine_results_not(self):
        results_sets = [
            [("doc1", 0.9, {}), ("doc2", 0.8, {})],
            [("doc2", 0.85, {}), ("doc3", 0.7, {})]
        ]
        operator = "NOT"
        expected = [("doc1", 0.9, {})]
        result = self.processor.combine_results(results_sets, operator)
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
