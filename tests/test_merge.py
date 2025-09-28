import pandas as pd

from phishing_benchmark.data.merge import merge_and_dedup


def test_merge_and_dedup_prefers_first_source():
    df1 = pd.DataFrame(
        {
            "id": ["a1"],
            "source_dataset": ["zefang-liu"],
            "subject": ["Hello"],
            "body_clean": ["test body"],
            "label": [1],
        }
    )
    df2 = pd.DataFrame(
        {
            "id": ["b1"],
            "source_dataset": ["cyradar"],
            "subject": ["Hello"],
            "body_clean": ["test body"],
            "label": [1],
        }
    )
    merged = merge_and_dedup([df1, df2])
    assert len(merged) == 1
    assert merged.iloc[0]["source_dataset"] == "zefang-liu"
