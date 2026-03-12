"""Tests for src/data/preprocess.py."""

import tempfile
from pathlib import Path

import pandas as pd

from src.data.preprocess import load_and_clean

SAMPLE_CSV = """Object ID,Title,Post Type,Author,Created At,URL,Points,Number of Comments
1,Test Post,story,alice,2010-01-01 12:00:00,http://example.com,42,5.0
2,Ask HN: Question,ask_hn,bob,2019-06-15 08:30:00,,10,3.0
3,Show HN: Demo,show_hn,carol,2022-03-20 14:00:00,http://demo.com,100,
1,Test Post,story,alice,2010-01-01 12:00:00,http://example.com,42,5.0
"""


def _load_sample() -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(SAMPLE_CSV)
        f.flush()
        return load_and_clean(Path(f.name))


def test_deduplication() -> None:
    df = _load_sample()
    assert len(df) == 3


def test_column_names() -> None:
    df = _load_sample()
    expected = [
        "object_id",
        "title",
        "post_type",
        "author",
        "created_at",
        "url",
        "points",
        "num_comments",
    ]
    assert list(df.columns) == expected


def test_null_num_comments_filled() -> None:
    df = _load_sample()
    assert df.loc[df["object_id"] == 3, "num_comments"].iloc[0] == 0


def test_types() -> None:
    df = _load_sample()
    assert df["object_id"].dtype == "int64"
    assert df["points"].dtype == "int64"
    assert df["num_comments"].dtype == "int64"
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
