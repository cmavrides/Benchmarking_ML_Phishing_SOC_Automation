import pandas as pd

from task_a_benchmark.preprocess import run_preprocess


def test_deduplication_and_stratified_split():
    data = pd.DataFrame(
        {
            "subject": ["Hello", "Hello", "Alert", "Update"],
            "body": ["Body", "Body", "Urgent", "Safe"],
            "body_is_html": [False, False, False, False],
            "label": [0, 0, 1, 1],
            "source_dataset": ["zefang_liu", "zefang_liu", "cyradar", "cyradar"],
            "source_type": ["email"] * 4,
        }
    )

    splits = run_preprocess(
        data,
        lower=True,
        strip_html_flag=False,
        min_length=1,
        max_length=100,
        val_size=0.25,
        test_size=0.25,
        seed=42,
        save_dir=None,
    )

    total_rows = sum(len(df) for df in splits.values())
    assert total_rows == 3  # duplicate removed
    labels = pd.concat([df["label"] for df in splits.values()])
    assert set(labels) == {0, 1}
