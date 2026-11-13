"""Run with: python -m unittest discover -s tests -v."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pandas.testing import assert_frame_equal
from PIL import Image

from utils.plotting import plot_adsorption


class PlotAdsorptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "nested" / "adsorption.png"
        self.results = pd.DataFrame({
            "mof_id": ["MOF-A"] * 4,
            "gas": ["CO2", "CO2", "N2", "N2"],
            "pressure": [1.0, 0.0, 1.0, 0.0],
            "adsorption": [4.0, -0.1, 0.8, 0.0],
        })
        self.database = pd.DataFrame({
            "mof_id": ["MOF-B"] * 4,
            "gas": ["CO2", "CO2", "N2", "N2"],
            "pressure": [1.0, 0.0, 1.0, 0.0],
            "adsorption": [3.5, 0.0, 0.6, 0.0],
        })

    def capture(self, database=None):
        """Inspect plotted values just before saving the real image."""
        panels = []
        savefig = Figure.savefig

        def save(figure, *args, **kwargs):
            for axis in figure.axes:
                panels.append({
                    "title": axis.get_title(),
                    "lines": {
                        line.get_label(): (
                            line.get_xdata().tolist(), line.get_ydata().tolist(),
                            line.get_linestyle(), line.get_color(),
                        ) for line in axis.lines
                    },
                })
            return savefig(figure, *args, **kwargs)

        with patch.object(Figure, "savefig", save):
            result = plot_adsorption(self.results, self.output, database)
        self.assertEqual(result, str(self.output))
        with Image.open(result) as image:
            image.verify()
        return panels

    def test_sorted_curves_preserve_zero_pressure_negative_prediction_and_inputs(self):
        original = self.results.copy(deep=True)
        db_original = self.database.copy(deep=True)
        panels = self.capture(self.database)
        self.assertEqual([p["title"] for p in panels], ["CO₂", "N₂"])
        self.assertEqual(panels[0]["lines"]["MOF-A (Prediction)"][:3],
                         ([0.0, 1.0], [-0.1, 4.0], "-"))
        self.assertEqual(panels[0]["lines"]["MOF-B (DB)"][:3],
                         ([0.0, 1.0], [0.0, 3.5], "--"))
        self.assertEqual(panels[0]["lines"]["MOF-A (Prediction)"][3],
                         panels[1]["lines"]["MOF-A (Prediction)"][3])
        assert_frame_equal(self.results, original)
        assert_frame_equal(self.database, db_original)

    def test_empty_database_is_optional(self):
        self.assertEqual(self.capture(), self.capture(pd.DataFrame()))

    def test_wide_database_matches_long_database(self):
        expected = self.capture(self.database)
        wide = self.database.pivot(index=["mof_id", "pressure"], columns="gas",
                                   values="adsorption").reset_index()
        wide = wide.rename(columns={"CO2": "CO2_adsorption", "N2": "N2_adsorption"})
        original = wide.copy(deep=True)
        self.assertEqual(self.capture(wide), expected)
        assert_frame_equal(wide, original)

    def test_long_database_aliases(self):
        aliased = self.database.rename(columns={
            "mof_id": "MOF_ID", "pressure": "Pressure",
            "gas": "Component", "adsorption": "Uptake",
        })
        self.assertEqual(self.capture(aliased), self.capture(self.database))

    def test_multiple_mofs_and_single_pressure_do_not_connect(self):
        self.results = pd.DataFrame({
            "mof_id": ["MOF-A", "MOF-B"], "gas": ["CO2", "CO2"],
            "pressure": [0.0, 1.0], "adsorption": [0.0, 3.0],
        })
        panels = self.capture()
        self.assertEqual(len(panels), 1)
        self.assertEqual(len(panels[0]["lines"]), 2)
        self.assertTrue(all(len(line[0]) == 1 for line in panels[0]["lines"].values()))

    def test_invalid_inputs_fail_before_creating_output(self):
        invalid = [
            self.results.iloc[:0], self.results.drop(columns="pressure"),
            self.results.assign(pressure=[np.nan, 0, 1, 0]),
            self.results.assign(adsorption=[np.inf, 0, 1, 0]),
            self.results.assign(adsorption=["invalid", 0, 1, 0]),
            self.results.assign(pressure=-1), self.results.assign(mof_id=None),
            self.results.assign(mof_id=" "), self.results.assign(gas="CH4"),
            pd.concat([self.results, self.results.iloc[:1]], ignore_index=True),
        ]
        for frame in invalid:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                plot_adsorption(frame, self.output)
        self.assertFalse(self.output.exists())
        with self.assertRaises(TypeError):
            plot_adsorption([], self.output)
        with self.assertRaises(TypeError):
            plot_adsorption(self.results, self.output, [])

    def test_invalid_database_is_reported_instead_of_silently_omitted(self):
        with self.assertRaisesRegex(ValueError, "db_results"):
            plot_adsorption(self.results, self.output, pd.DataFrame({"mof_id": ["A"]}))
        with self.assertRaisesRegex(ValueError, "measurement condition"):
            plot_adsorption(self.results, self.output,
                            pd.concat([self.database, self.database], ignore_index=True))

    def test_different_temperatures_with_disjoint_pressures_are_rejected(self):
        long = self.database.assign(temperature=[298, 300, 298, 300])
        wide = pd.DataFrame({
            "mof_id": ["MOF-B", "MOF-B"], "pressure": [0, 1],
            "temperature": [298, 300], "CO2": [0, 3], "N2": [0, 0.5],
        })
        for database in (long, wide):
            with self.subTest(database=database), self.assertRaisesRegex(ValueError, "temperature"):
                plot_adsorption(self.results, self.output, database)

    def test_formats_and_exact_returned_path(self):
        for suffix in ("PNG", "jpg", "jpeg", "svg", "pdf"):
            with self.subTest(suffix=suffix):
                path = Path(self.temp.name) / f"plot.{suffix}"
                self.assertEqual(plot_adsorption(self.results, str(path)), str(path))
                self.assertGreater(path.stat().st_size, 100)
        with self.assertRaisesRegex(ValueError, "output_path"):
            plot_adsorption(self.results, self.output.with_suffix(""))

    def test_save_failure_releases_figure(self):
        figures = []

        def fail(figure, *args, **kwargs):
            figures.append(figure)
            raise OSError("disk error")

        with patch.object(Figure, "savefig", fail), self.assertRaises(OSError):
            plot_adsorption(self.results, self.output)
        self.assertEqual(figures[0].axes, [])


if __name__ == "__main__":
    unittest.main()
