"""Verify completeness of the ForecastIQ model release."""

import unittest
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "releases/forecastiq-models-v2.zip"


class TestReleaseArtifacts(unittest.TestCase):

    def test_archive_completeness(self):
        self.assertTrue(ARCHIVE.exists(), "Release archive missing")

        with ZipFile(ARCHIVE) as archive:
            names = set(archive.namelist())

            required = [
                "validation/models_v2/xgboost_model.joblib",
                "validation/models_v2/encoders.joblib",
                "validation/models_v2/features.joblib",
                "models/multiseries/lstm/global_lstm.pt",
                "models/multiseries/lstm/global_lstm_config.pkl",
                "models/multiseries/lstm/v2/production/global_lstm_v2.pt",
                "models/multiseries/lstm/v2/production/global_lstm_v2_config.pkl",
                "models/multiseries/lstm/v2/validation_analysis/uncertainty_calibration.csv",
            ]

            for filename in required:
                with self.subTest(file=filename):
                    self.assertIn(filename, names)

            for model in ("sarima", "prophet"):
                for store in range(1, 6):
                    for item in range(1, 11):
                        filename = (
                            f"models/multiseries/{model}/"
                            f"store_{store}_ITEM_{item:03d}.pkl"
                        )
                        with self.subTest(file=filename):
                            self.assertIn(filename, names)


if __name__ == "__main__":
    unittest.main()
