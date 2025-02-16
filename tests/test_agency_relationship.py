import unittest
from scripts.search_regulations import AgencyRelationshipMapper

class TestAgencyRelationshipMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = AgencyRelationshipMapper()

    def test_get_related_agencies_parent(self):
        agency = "irs"
        expected = {
            "parent": ["department-of-treasury"],
            "child": ["irs", "fiscal-service"],
            "collaborator": ["federal-reserve"],
            "delegated_authority": ["state-banking-departments"]
        }
        result = self.mapper.get_related_agencies(agency)
        self.assertEqual(result["parent"], expected["parent"])

    def test_get_related_agencies_child(self):
        agency = "department-of-treasury"
        expected = {
            "parent": ["department-of-treasury"],
            "child": ["irs", "fiscal-service"],
            "collaborator": ["federal-reserve"],
            "delegated_authority": ["state-banking-departments"]
        }
        result = self.mapper.get_related_agencies(agency)
        self.assertEqual(result["child"], expected["child"])

    def test_get_related_agencies_collaborator(self):
        agency = "federal-reserve"
        expected = {
            "parent": ["department-of-treasury"],
            "child": ["irs", "fiscal-service"],
            "collaborator": ["federal-reserve"],
            "delegated_authority": ["state-banking-departments"]
        }
        result = self.mapper.get_related_agencies(agency)
        self.assertEqual(result["collaborator"], expected["collaborator"])

    def test_get_related_agencies_delegated_authority(self):
        agency = "state-banking-departments"
        expected = {
            "parent": ["department-of-treasury"],
            "child": ["irs", "fiscal-service"],
            "collaborator": ["federal-reserve"],
            "delegated_authority": ["state-banking-departments"]
        }
        result = self.mapper.get_related_agencies(agency)
        self.assertEqual(result["delegated_authority"], expected["delegated_authority"])

if __name__ == "__main__":
    unittest.main()
