from pathlib import Path

import pandas as pd

from task_a_benchmark import loaders


def test_loaders_map_columns(tmp_path: Path):
    zefang_path = tmp_path / "zefang_liu.csv"
    zefang_df = pd.DataFrame(
        {
            "Email_Text": ["legit email", "phish"],
            "Category": ["legitimate", "phishing"],
            "subject": ["Hello", "Alert"],
        }
    )
    zefang_df.to_csv(zefang_path, index=False)

    cyradar_path = tmp_path / "cyradar.csv"
    cyradar_df = pd.DataFrame(
        {
            "text": ["safe", "danger"],
            "label": [0, 1],
        }
    )
    cyradar_df.to_csv(cyradar_path, index=False)

    df = loaders.load_datasets(tmp_path, use_zefang=True, use_cyradar=True)
    assert set(df.columns) == {"subject", "body", "body_is_html", "label", "source_dataset", "source_type"}
    assert sorted(df["source_dataset"].unique()) == ["cyradar", "zefang_liu"]
    assert df["label"].isin({0, 1}).all()
