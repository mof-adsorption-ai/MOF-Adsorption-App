import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.db_search import search_mofs


class SearchMofsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "isotherms.parquet"
        self.data = pd.DataFrame({
            "mof_id": [1, 2, 3],
            "mof_name": ["A", "B", "C"],
            "co2_pressure": [[0.1, 0.5]] * 3,
            "co2_loading": [[4., 8.], [6., 10.], [1., 2.]],
            "co2_temperature_K": [[298., 298.]] * 3,
            "n2_pressure": [[0.1]] * 3,
            "n2_loading": [[1.], [2.], [0.]],
            "n2_temperature_K": [[298.]] * 3,
        })
        self.data.to_parquet(self.path)

    def search(self, gas, pressure=0.1, extremum="max", count=2):
        return search_mofs(gas, pressure, extremum, count, db_path=self.path)

    def test_array_layout_and_ranking(self):
        self.assertEqual(self.search("CO2").mof_id.tolist(), [2, 1])
        self.assertEqual(self.search("CO2", extremum="min").mof_id.tolist(), [3, 1])
        self.assertTrue(self.search("CO2", pressure=0.2).empty)
        self.assertEqual(self.search("N2").mof_id.tolist(), [2, 1])

    def test_ratio_and_measurement_conditions(self):
        self.assertEqual(self.search("CO2/N2").mof_id.tolist(), [1, 2])
        self.data["n2_pressure"] = [[0.09]] * 3
        self.data.to_parquet(self.path)
        self.assertTrue(self.search("CO2/N2").empty)
        self.data["n2_pressure"] = [[0.1]] * 3
        self.data["n2_temperature_K"] = [[300.]] * 3
        self.data.to_parquet(self.path)
        self.assertTrue(self.search("CO2/N2").empty)

    def test_wide_layout_and_invalid_inputs(self):
        pd.DataFrame({"mof_id": [1, 2], "pressure": [0.1, 0.1], "CO2": [2., 4.], "N2": [1., 1.]}).to_parquet(self.path)
        self.assertEqual(self.search("CO2/N2").mof_id.tolist(), [2, 1])
        for args in [("O2", 0.1, "max", 2), ("CO2", -1, "max", 2), ("CO2", 0.1, "max", 0)]:
            with self.assertRaises(ValueError):
                self.search(*args)


if __name__ == "__main__":
    unittest.main()
