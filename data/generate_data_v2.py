"""Generate reproducible synthetic retail sales for ForecastIQ."""

from pathlib import Path

import numpy as np
import pandas as pd


START_DATE = "2017-01-01"
END_DATE = "2020-12-31"
SEED = 42

STORE_TYPES = {
    1: "A",
    2: "A",
    3: "B",
    4: "B",
    5: "C",
}

CATEGORIES = [
    "GROCERY",
    "GROCERY",
    "BEVERAGES",
    "BEVERAGES",
    "CLEANING",
    "CLEANING",
    "PRODUCE",
    "PRODUCE",
    "DAIRY",
    "DAIRY",
]


def generate_retail_data(seed=SEED):
    rng = np.random.default_rng(seed)

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    stores = pd.DataFrame({
        "store_nbr": list(STORE_TYPES),
        "type": list(STORE_TYPES.values()),
        "cluster": [1, 1, 2, 2, 3],
    })

    records = []

    for store_id, store_type in STORE_TYPES.items():
        store_multiplier = {
            "A": 1.3,
            "B": 1.0,
            "C": 0.7,
        }[store_type]

        for product_index, category in enumerate(
            CATEGORIES, start=1
        ):
            product = f"ITEM_{product_index:03d}"

            base_demand = rng.uniform(20, 200)
            trend_rate = rng.uniform(0.005, 0.02)

            for day_index, current_date in enumerate(dates):
                weekend = (
                    1.3 if current_date.dayofweek >= 5
                    else 1.0
                )

                annual = (
                    1.0
                    + 0.3 * np.sin(
                        2 * np.pi
                        * current_date.dayofyear / 365
                    )
                )

                holiday = 1.0

                if (
                    current_date.month == 12
                    and current_date.day >= 20
                ):
                    holiday = 1.6
                elif (
                    current_date.month == 1
                    and current_date.day <= 3
                ):
                    holiday = 1.4

                promotion = int(rng.random() < 0.2)

                promotion_effect = (
                    1.25 if promotion else 1.0
                )

                noise = rng.normal(1.0, 0.1)

                demand = max(
                    0.0,
                    (base_demand + trend_rate * day_index)
                    * store_multiplier
                    * weekend
                    * annual
                    * holiday
                    * promotion_effect
                    * noise,
                )

                records.append({
                    "date": current_date.date(),
                    "store_nbr": store_id,
                    "item_nbr": product,
                    "family": category,
                    "sales": round(demand, 2),
                    "onpromotion": promotion,
                })

    sales = pd.DataFrame.from_records(records)

    return sales, stores


if __name__ == "__main__":
    output = Path("validation/generated_v2")
    output.mkdir(parents=True, exist_ok=True)

    sales, stores = generate_retail_data()

    sales.to_csv(output / "train.csv", index=False)
    stores.to_csv(output / "stores.csv", index=False)

    print("Independent dataset generated")
    print(f"Observations: {len(sales):,}")
    print(f"Stores: {sales['store_nbr'].nunique()}")
    print(f"Products: {sales['item_nbr'].nunique()}")
    print(f"Output directory: {output}")
