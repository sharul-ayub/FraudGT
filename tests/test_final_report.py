import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fraudGT.final_report import generate_final_report


class FinalReportTest(unittest.TestCase):
    def test_generates_binary_metrics_tables_and_plots(self):
        true = torch.tensor([0, 0, 1, 1])
        probabilities = torch.tensor(
            [[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.1, 0.9]]
        )
        log_probabilities = probabilities.log()

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            metrics = generate_final_report(
                true=true,
                pred_score=log_probabilities,
                output_dir=output,
                epoch=7,
                validation_f1=0.75,
                test_loss=0.25,
            )

            self.assertEqual(metrics["best_epoch"], 7)
            self.assertEqual(metrics["f1"], 0.5)
            self.assertAlmostEqual(metrics["roc_auc"], 0.75)
            self.assertAlmostEqual(metrics["pr_auc"], 0.7916666667)
            self.assertAlmostEqual(metrics["average_precision"], 0.8333333333)

            expected_files = {
                "metrics.json",
                "confusion_matrix.csv",
                "predictions.csv",
                "confusion_matrix.png",
                "roc_curve.png",
                "precision_recall_curve.png",
            }
            self.assertEqual(expected_files, {path.name for path in output.iterdir()})
            self.assertEqual(json.loads((output / "metrics.json").read_text()), metrics)

            with (output / "confusion_matrix.csv").open(newline="") as handle:
                matrix = next(csv.DictReader(handle))
            self.assertEqual(
                matrix,
                {
                    "true_negative": "1",
                    "false_positive": "1",
                    "false_negative": "1",
                    "true_positive": "1",
                },
            )

            with (output / "predictions.csv").open(newline="") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), 4)
            self.assertAlmostEqual(float(predictions[0]["fraud_probability"]), 0.1)

    def test_filters_unknown_labels_and_handles_single_known_class(self):
        true = torch.tensor([-1, 0, 0])
        probabilities = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])

        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics = generate_final_report(
                true=true,
                pred_score=probabilities.log(),
                output_dir=temporary_directory,
                epoch=1,
                validation_f1=0.2,
                test_loss=0.5,
            )

            self.assertEqual(metrics["num_test_examples"], 2)
            self.assertIsNone(metrics["roc_auc"])
            self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
