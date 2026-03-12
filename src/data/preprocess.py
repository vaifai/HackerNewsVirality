"""Step 1: Load raw hn.csv, clean, type-cast, and save as parquet."""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


COLUMN_RENAME = {
    "Object ID": "object_id",
    "Title": "title",
    "Post Type": "post_type",
    "Author": "author",
    "Created At": "created_at",
    "URL": "url",
    "Points": "points",
    "Number of Comments": "num_comments",
}

DTYPES = {
    "Object ID": "int64",
    "Title": "str",
    "Post Type": "str",
    "Author": "str",
    "URL": "str",
    "Points": "int64",
}


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        dtype=DTYPES,
        parse_dates=["Created At"],
    )
    df.rename(columns=COLUMN_RENAME, inplace=True)

    # num_comments: float (has NaN) → fill with 0, cast to int
    df["num_comments"] = df["num_comments"].fillna(0).astype("int64")

    # Drop rows with null title or points (unusable)
    null_title = df["title"].isna().sum()
    null_points = df["points"].isna().sum()
    if null_title > 0:
        print(f"Dropping {null_title} rows with null title")
    if null_points > 0:
        print(f"Dropping {null_points} rows with null points")
    df.dropna(subset=["title", "points"], inplace=True)

    # Drop exact duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    dupes = before - len(df)
    if dupes > 0:
        print(f"Dropped {dupes} exact duplicate rows")

    return df


def validate(df: pd.DataFrame) -> None:
    assert df["object_id"].notna().all(), "Found null object_id"
    assert df["title"].notna().all(), "Found null title after cleaning"
    assert df["points"].notna().all(), "Found null points after cleaning"

    min_year = df["created_at"].dt.year.min()
    max_year = df["created_at"].dt.year.max()
    assert min_year == 2006, f"Expected min year 2006, got {min_year}"
    assert max_year == 2023, f"Expected max year 2023, got {max_year}"


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nRows: {len(df):,}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nDate range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"Points — median: {df['points'].median()}, mean: {df['points'].mean():.1f}")
    print(f"\nPost types:\n{df['post_type'].value_counts()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw HN CSV")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "hn.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "hn_cleaned.parquet")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = load_and_clean(args.input)

    print("Validating...")
    validate(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False, engine="pyarrow")
    print(f"Saved to {args.output}")

    print_summary(df)


if __name__ == "__main__":
    main()
