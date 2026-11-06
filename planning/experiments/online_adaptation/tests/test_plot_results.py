"""Headless plotting checks using explicit synthetic metric fixtures only."""

import json
import tempfile
import unittest
from pathlib import Path

from planning.experiments.online_adaptation.plot_results import load_rounds, make_figure, plot_results


class PlotResultsTests(unittest.TestCase):
    def record(self, round_number, planning=None):
        return {
            "round": round_number,
            "experience_transitions": round_number * 200,
            "updates": round_number * 10,
            "target": {
                "diagnostics": {
                    "representation.linear_cka": 1 - round_number * 0.03,
                    "representation.relative_coordinate_drift": round_number * 0.1,
                    "horizon_5.prediction.block_position_rmse": 25 - round_number,
                    "horizon_5.persistence.block_position_rmse": 30,
                    "observed.block_position_rmse": 5,
                    "horizon_5.prediction.block_angle_mae_rad": 0.4,
                    "horizon_5.persistence.block_angle_mae_rad": 0.5,
                },
                "planning": planning,
            },
            "source": {}, "training": {}, "timing": {},
        }

    def test_real_values_and_missing_planning_without_invented_curves(self):
        records = [self.record(0), self.record(1), self.record(2, {"success_rate": 37.5})]
        records[1]["target"]["diagnostics"]["representation.linear_cka"] = None
        figure = make_figure(records)
        self.assertEqual(len(figure.axes), 6)
        cka = figure.axes[0].lines[0]
        self.assertEqual(cka.get_xdata().tolist(), [0, 400])
        self.assertEqual(cka.get_ydata().tolist(), [1, 0.94])
        control = figure.axes[4].lines[0]
        self.assertEqual(control.get_xdata().tolist(), [400])
        self.assertEqual(control.get_ydata().tolist(), [37.5])
        self.assertEqual(len(figure.axes[2].lines), 2)
        self.assertEqual(len(figure.axes[4].lines), 1)
        figure.clear()

    def test_missing_planning_panel_has_no_lines(self):
        figure = make_figure([self.record(0)])
        self.assertEqual(len(figure.axes[4].lines), 0)
        self.assertIn("No recorded measurements", [text.get_text() for text in figure.axes[4].texts])
        figure.clear()

    def test_round_order_and_pdf_export_without_display(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metrics").mkdir()
            for number in (10, 2):
                (root / "metrics" / f"round_{number}.json").write_text(json.dumps(self.record(number)))
            self.assertEqual([item["round"] for item in load_rounds(root)], [2, 10])
            output = plot_results(root)
            self.assertEqual(output, root / "adaptation_curves.pdf")
            self.assertTrue(output.read_bytes().startswith(b"%PDF"))

    def test_empty_results_raise_without_creating_a_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                plot_results(temporary)
            self.assertFalse((Path(temporary) / "adaptation_curves.pdf").exists())


if __name__ == "__main__":
    unittest.main()
