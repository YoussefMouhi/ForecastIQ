"""Restore ForecastIQ model artifacts from a local release archive."""

from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "releases" / "forecastiq-models-v2.zip"

REQUIRED = [
    "validation/models_v2/xgboost_model.joblib",
    "validation/models_v2/encoders.joblib",
    "validation/models_v2/features.joblib",
    "models/multiseries/lstm/global_lstm.pt",
    "models/multiseries/lstm/global_lstm_config.pkl",
    "models/multiseries/lstm/v2/production/global_lstm_v2.pt",
    "models/multiseries/lstm/v2/production/global_lstm_v2_config.pkl",
    "models/multiseries/lstm/v2/validation_analysis/uncertainty_calibration.csv",
]

def main():
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Model archive not found: {ARCHIVE}")

    with ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError("Model archive failed integrity verification")

        members = archive.namelist()

        missing = [path for path in REQUIRED if path not in members]
        if missing:
            raise ValueError(f"Archive missing required artifacts: {missing}")

        for model in ["sarima", "prophet"]:
            count = sum(
                name.startswith(f"models/multiseries/{model}/store_")
                and name.endswith(".pkl")
                for name in members
            )
            if count != 50:
                raise ValueError(f"Expected 50 {model} models, found {count}")

        for info in archive.infolist():
            path = Path(info.filename)

            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive entry: {info.filename}")

            if info.is_dir():
                continue

            destination = ROOT / path
            destination.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(info) as source:
                # Never overwrite an existing trained model.
                if destination.exists():
                    continue
                with destination.open("xb") as target:
                    import shutil
                    shutil.copyfileobj(source, target)

    print("ForecastIQ model installation completed.")

if __name__ == "__main__":
    main()
