a"""
Crypto Telegram Channels Analysis
=================================

This script prepares and analyzes Telegram cryptocurrency channel data.

Main tasks:
1. Load and clean Telegram post-level data
2. Convert Jalali dates to Gregorian dates
3. Download daily cryptocurrency market data from Yahoo Finance
4. Calculate engagement rate
5. Merge Telegram data with price and volume data
6. Run descriptive, correlation, timing, and event-study analyses
7. Export clean datasets, tables, and figures for dashboard/reporting

Expected input file:
    data/raw/Project2_Dataset.csv

Example:
    python src/crypto_telegram_analysis.py --input data/raw/Project2_Dataset.csv

Author:
    Sarah Azizi
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional

import jdatetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CRYPTO_SYMBOLS: Dict[str, str] = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "binancecoin": "BNB-USD",
    "cardano": "ADA-USD",
    "tether": "USDT-USD",
}

REQUIRED_COLUMNS = {
    "channel_id",
    "date",
    "content_type",
    "crypto_mentioned",
    "views",
    "forwards",
    "member_count",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def ensure_directories(*paths: Path) -> None:
    """Create output folders if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def normalize_crypto_name(value: object) -> str:
    """Standardize cryptocurrency names for merging with market data."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    mapping = {
        "btc": "bitcoin",
        "bitcoin": "bitcoin",
        "eth": "ethereum",
        "ethereum": "ethereum",
        "bnb": "binancecoin",
        "binance": "binancecoin",
        "binance coin": "binancecoin",
        "binancecoin": "binancecoin",
        "ada": "cardano",
        "cardano": "cardano",
        "usdt": "tether",
        "tether": "tether",
    }

    return mapping.get(text, text)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Check whether the input dataset contains the required columns."""
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The input file is missing these required columns: "
            + ", ".join(sorted(missing_columns))
        )


def read_data(input_path: Path) -> pd.DataFrame:
    """Read CSV or Excel input data."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Input file must be CSV or Excel.")

    df.columns = df.columns.str.strip()
    validate_input_columns(df)

    return df


def jalali_to_gregorian(value: object) -> pd.Timestamp:
    """
    Convert a Jalali date string to a Gregorian pandas Timestamp.

    Accepted formats:
    - 1403-01-15
    - 1403/01/15
    """
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip().replace("-", "/")

    try:
        gregorian_date = jdatetime.datetime.strptime(text, "%Y/%m/%d").togregorian()
        return pd.to_datetime(gregorian_date.date())
    except Exception:
        return pd.NaT


# ---------------------------------------------------------------------
# Data cleaning and preprocessing
# ---------------------------------------------------------------------

def clean_telegram_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Telegram channel data and create date/engagement features.
    """
    df = df.copy()

    # Standardize crypto names
    df["crypto_mentioned"] = df["crypto_mentioned"].apply(normalize_crypto_name)

    # Convert numeric columns
    numeric_columns = ["views", "forwards", "member_count"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Correct impossible negative values
    df["views"] = df["views"].abs()
    df["forwards"] = df["forwards"].abs()
    df["member_count"] = df["member_count"].abs()

    # Fill missing member_count using channel-level median, then global median
    df["member_count"] = df.groupby("channel_id")["member_count"].transform(
        lambda s: s.fillna(s.median())
    )
    df["member_count"] = df["member_count"].fillna(df["member_count"].median())

    # Replace zero member_count to avoid division by zero
    df.loc[df["member_count"] == 0, "member_count"] = np.nan

    # Use date_fixed if it exists; otherwise use date
    source_date_col = "date_fixed" if "date_fixed" in df.columns else "date"
    df["date_jalali_clean"] = df[source_date_col].astype(str).str.strip().str.replace("-", "/", regex=False)
    df["date_miladi"] = df["date_jalali_clean"].apply(jalali_to_gregorian)

    # Drop rows with invalid dates
    df = df.dropna(subset=["date_miladi"]).copy()

    # Date features
    df["year"] = df["date_miladi"].dt.year
    df["month"] = df["date_miladi"].dt.month
    df["day"] = df["date_miladi"].dt.day
    df["day_of_week"] = df["date_miladi"].dt.day_name()
    df["month_name"] = df["date_miladi"].dt.month_name()

    # Engagement Rate: views + forwards divided by channel member count
    df["engagement_rate"] = (df["views"] + df["forwards"]) / df["member_count"]

    return df


def generate_data_quality_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Create a short data-quality summary table."""
    report = pd.DataFrame(
        {
            "metric": [
                "raw_rows",
                "raw_columns",
                "clean_rows",
                "clean_columns",
                "invalid_or_removed_rows",
                "missing_values_after_cleaning",
                "unique_channels",
                "unique_cryptos",
            ],
            "value": [
                raw_df.shape[0],
                raw_df.shape[1],
                clean_df.shape[0],
                clean_df.shape[1],
                raw_df.shape[0] - clean_df.shape[0],
                int(clean_df.isna().sum().sum()),
                clean_df["channel_id"].nunique(),
                clean_df["crypto_mentioned"].nunique(),
            ],
        }
    )

    return report


# ---------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------

def download_market_data(
    symbols: Dict[str, str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download OHLCV market data from Yahoo Finance and calculate:
    - Typical Price
    - Price Change %
    - Trade Volume
    """
    all_market_data = []

    for crypto_name, ticker in symbols.items():
        print(f"Downloading market data for {crypto_name} ({ticker})...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )

        if df.empty:
            print(f"Warning: No market data downloaded for {crypto_name}.")
            continue

        # yfinance may return MultiIndex columns in some versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["date_miladi"] = pd.to_datetime(df["Date"]).dt.normalize()
        df["crypto_mentioned"] = crypto_name

        df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["price_change_pct"] = df["typical_price"].pct_change() * 100
        df["trade_volume"] = df["Volume"]

        all_market_data.append(
            df[
                [
                    "crypto_mentioned",
                    "date_miladi",
                    "typical_price",
                    "price_change_pct",
                    "trade_volume",
                ]
            ]
        )

    if not all_market_data:
        raise RuntimeError("No market data was downloaded. Check your internet connection or ticker symbols.")

    market_data = pd.concat(all_market_data, ignore_index=True)

    return market_data


def merge_telegram_with_market_data(
    telegram_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Telegram post data with daily price and volume data."""
    merged = telegram_df.merge(
        market_df,
        on=["crypto_mentioned", "date_miladi"],
        how="left",
    )

    return merged


# ---------------------------------------------------------------------
# Descriptive analysis and figures
# ---------------------------------------------------------------------

def save_categorical_charts(df: pd.DataFrame, output_dir: Path) -> None:
    """Save bar charts for categorical variables."""
    categorical_columns = ["content_type", "crypto_mentioned"]

    if "macro_event" in df.columns:
        categorical_columns.append("macro_event")

    for col in categorical_columns:
        plt.figure(figsize=(9, 5))
        df[col].value_counts().head(10).plot(kind="bar")
        plt.title(f"Top categories: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_distribution.png", dpi=150)
        plt.close()


def save_numeric_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Save boxplot for key numeric columns."""
    numeric_columns = ["views", "forwards", "member_count", "engagement_rate"]
    available_columns = [col for col in numeric_columns if col in df.columns]

    plt.figure(figsize=(10, 6))
    df[available_columns].plot(kind="box", subplots=True, layout=(2, 2), figsize=(12, 8))
    plt.suptitle("Box Plot for Numeric Features")
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_features_boxplot.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Correlation and timing analysis
# ---------------------------------------------------------------------

def safe_spearman(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Calculate Spearman correlation safely."""
    subset = df[[col1, col2]].dropna()

    if subset.shape[0] < 3:
        return np.nan

    return subset.corr(method="spearman").iloc[0, 1]


def correlation_by_crypto(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman correlations by cryptocurrency."""
    result = (
        df.groupby("crypto_mentioned")
        .apply(
            lambda x: pd.Series(
                {
                    "spearman_views_vs_price_change": safe_spearman(x, "views", "price_change_pct"),
                    "spearman_forwards_vs_price_change": safe_spearman(x, "forwards", "price_change_pct"),
                    "spearman_members_vs_price_change": safe_spearman(x, "member_count", "price_change_pct"),
                    "spearman_engagement_vs_price_change": safe_spearman(x, "engagement_rate", "price_change_pct"),
                }
            )
        )
        .reset_index()
    )

    return result


def correlation_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman correlations by Telegram channel."""
    result = (
        df.groupby("channel_id")
        .apply(
            lambda x: pd.Series(
                {
                    "spearman_views_vs_price_change": safe_spearman(x, "views", "price_change_pct"),
                    "spearman_forwards_vs_price_change": safe_spearman(x, "forwards", "price_change_pct"),
                    "spearman_engagement_vs_price_change": safe_spearman(x, "engagement_rate", "price_change_pct"),
                }
            )
        )
        .reset_index()
    )

    return result


def engagement_by_time(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Summarize engagement rate by weekday and month."""
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    by_weekday = (
        df.groupby("day_of_week")
        .agg(
            avg_engagement_rate=("engagement_rate", "mean"),
            post_count=("engagement_rate", "count"),
        )
        .reindex(weekday_order)
        .reset_index()
    )

    by_month = (
        df.groupby("month_name")
        .agg(
            avg_engagement_rate=("engagement_rate", "mean"),
            post_count=("engagement_rate", "count"),
        )
        .reindex(month_order)
        .reset_index()
    )

    return {
        "engagement_by_weekday": by_weekday,
        "engagement_by_month": by_month,
    }


def top_engagement_channels(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank channels by average engagement rate."""
    result = (
        df.groupby("channel_id")
        .agg(
            avg_engagement_rate=("engagement_rate", "mean"),
            post_count=("engagement_rate", "count"),
            avg_member_count=("member_count", "mean"),
        )
        .sort_values("avg_engagement_rate", ascending=False)
        .head(top_n)
        .reset_index()
    )

    return result


def engagement_by_content_type(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate engagement rate by content category."""
    result = (
        df.groupby("content_type")
        .agg(
            avg_engagement_rate=("engagement_rate", "mean"),
            post_count=("engagement_rate", "count"),
        )
        .sort_values("avg_engagement_rate", ascending=False)
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# Event study: AAR and Hit Ratio
# ---------------------------------------------------------------------

def compute_event_study(
    events_df: pd.DataFrame,
    market_df: pd.DataFrame,
    window_days: int = 3,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate Abnormal Return (AR) for Telegram post events.

    AR = average price change after the event - average price change before the event
    Hit = 1 if abs(AR) > threshold, otherwise 0
    """
    results = []

    market_df = market_df.dropna(subset=["price_change_pct"]).copy()

    for _, row in events_df.iterrows():
        channel = row["channel_id"]
        crypto = row["crypto_mentioned"]
        event_date = row["date_miladi"]

        subset = market_df[market_df["crypto_mentioned"] == crypto]

        before = subset[
            (subset["date_miladi"] < event_date)
            & (subset["date_miladi"] >= event_date - pd.Timedelta(days=window_days))
        ]["price_change_pct"].mean()

        after = subset[
            (subset["date_miladi"] > event_date)
            & (subset["date_miladi"] <= event_date + pd.Timedelta(days=window_days))
        ]["price_change_pct"].mean()

        if pd.isna(before) or pd.isna(after):
            continue

        abnormal_return = after - before
        hit = int(abs(abnormal_return) > threshold)

        results.append(
            {
                "crypto_mentioned": crypto,
                "channel_id": channel,
                "event_date": event_date,
                "price_change_before": before,
                "price_change_after": after,
                "abnormal_return": abnormal_return,
                "hit": hit,
            }
        )

    return pd.DataFrame(results)


def summarize_aar_by_channel(
    event_results: pd.DataFrame,
    source_df: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """Summarize AAR and Hit Ratio by channel."""
    if event_results.empty:
        return pd.DataFrame()

    avg_members = (
        source_df.groupby("channel_id")["member_count"]
        .mean()
        .reset_index()
        .rename(columns={"member_count": "avg_member_count"})
    )

    summary = (
        event_results.groupby("channel_id")
        .agg(
            AAR=("abnormal_return", "mean"),
            signal_count=("abnormal_return", "count"),
            hit_ratio=("hit", "mean"),
        )
        .reset_index()
        .merge(avg_members, on="channel_id", how="left")
    )

    summary = summary.rename(
        columns={
            "AAR": f"AAR_{window_days}days",
            "signal_count": f"signal_count_{window_days}d",
            "hit_ratio": f"hit_ratio_{window_days}d",
        }
    )

    summary[f"hit_ratio_{window_days}d"] *= 100
    summary = summary.sort_values(f"AAR_{window_days}days", ascending=False)

    return summary


def summarize_top_aar_by_crypto_channel(
    event_results: pd.DataFrame,
    window_days: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top channels by AAR for each cryptocurrency."""
    if event_results.empty:
        return pd.DataFrame()

    summary = (
        event_results.groupby(["crypto_mentioned", "channel_id"])
        .agg(
            AAR=("abnormal_return", "mean"),
            signal_count=("abnormal_return", "count"),
            hit_ratio=("hit", "mean"),
        )
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "AAR": f"AAR_{window_days}days",
            "signal_count": f"signal_count_{window_days}d",
            "hit_ratio": f"hit_ratio_{window_days}d",
        }
    )

    summary[f"hit_ratio_{window_days}d"] *= 100

    top_channels = (
        summary.sort_values(
            ["crypto_mentioned", f"AAR_{window_days}days"],
            ascending=[True, False],
        )
        .groupby("crypto_mentioned")
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_channels


def run_event_study_outputs(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
    output_dir: Path,
    signal_only: bool = False,
    threshold: float = 1.0,
) -> None:
    """Run event-study analysis for 3-day and 5-day windows and save Excel outputs."""
    source_df = df.copy()

    if signal_only:
        source_df = source_df[source_df["content_type"].astype(str).str.strip() == "سیگنال"].copy()
        output_file = output_dir / "signal_only_AAR_hit_ratio_results.xlsx"
    else:
        output_file = output_dir / "all_posts_AAR_hit_ratio_results.xlsx"

    events = (
        source_df[["channel_id", "crypto_mentioned", "date_miladi"]]
        .drop_duplicates()
        .sort_values(["channel_id", "date_miladi"])
        .reset_index(drop=True)
    )

    event_results_3d = compute_event_study(
        events,
        market_df,
        window_days=3,
        threshold=threshold,
    )

    event_results_5d = compute_event_study(
        events,
        market_df,
        window_days=5,
        threshold=threshold,
    )

    aar_by_channel_3d = summarize_aar_by_channel(event_results_3d, source_df, window_days=3)
    aar_by_channel_5d = summarize_aar_by_channel(event_results_5d, source_df, window_days=5)

    top_by_crypto_3d = summarize_top_aar_by_crypto_channel(event_results_3d, window_days=3)
    top_by_crypto_5d = summarize_top_aar_by_crypto_channel(event_results_5d, window_days=5)

    with pd.ExcelWriter(output_file) as writer:
        event_results_3d.to_excel(writer, sheet_name="event_level_3d", index=False)
        event_results_5d.to_excel(writer, sheet_name="event_level_5d", index=False)
        aar_by_channel_3d.to_excel(writer, sheet_name="channel_AAR_3d", index=False)
        aar_by_channel_5d.to_excel(writer, sheet_name="channel_AAR_5d", index=False)
        top_by_crypto_3d.to_excel(writer, sheet_name="top_crypto_channel_3d", index=False)
        top_by_crypto_5d.to_excel(writer, sheet_name="top_crypto_channel_5d", index=False)

    print(f"Saved event-study output: {output_file}")


# ---------------------------------------------------------------------
# Export analysis outputs
# ---------------------------------------------------------------------

def save_analysis_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Save key analytical tables for dashboard and report."""
    tables = {
        "correlation_by_crypto.xlsx": correlation_by_crypto(df),
        "correlation_by_channel.xlsx": correlation_by_channel(df),
        "top_engagement_channels.xlsx": top_engagement_channels(df),
        "engagement_by_content_type.xlsx": engagement_by_content_type(df),
    }

    time_tables = engagement_by_time(df)
    for filename, table in tables.items():
        table.to_excel(output_dir / filename, index=False)

    for name, table in time_tables.items():
        table.to_excel(output_dir / f"{name}.xlsx", index=False)


def export_dashboard_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Export a clean dataset for Power BI or other dashboards."""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    output_base: Path,
    market_start: Optional[str] = None,
    market_end: Optional[str] = None,
    threshold: float = 1.0,
) -> None:
    """Run the full analysis pipeline."""
    processed_dir = output_base / "data" / "processed"
    figures_dir = output_base / "reports" / "figures"
    tables_dir = output_base / "reports" / "tables"

    ensure_directories(processed_dir, figures_dir, tables_dir)

    print("Reading Telegram data...")
    raw_df = read_data(input_path)

    print("Cleaning Telegram data...")
    clean_df = clean_telegram_data(raw_df)

    quality_report = generate_data_quality_report(raw_df, clean_df)
    quality_report.to_excel(tables_dir / "data_quality_report.xlsx", index=False)

    clean_df.to_csv(processed_dir / "telegram_cleaned.csv", index=False, encoding="utf-8-sig")

    # Define market data date range from Telegram data if not provided
    if market_start is None:
        market_start = (clean_df["date_miladi"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    if market_end is None:
        market_end = (clean_df["date_miladi"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    print(f"Market data range: {market_start} to {market_end}")

    print("Downloading market data...")
    market_df = download_market_data(CRYPTO_SYMBOLS, start_date=market_start, end_date=market_end)
    market_df.to_csv(processed_dir / "crypto_market_data.csv", index=False, encoding="utf-8-sig")

    print("Merging Telegram and market data...")
    merged_df = merge_telegram_with_market_data(clean_df, market_df)
    merged_df.to_excel(processed_dir / "telegram_with_market_data.xlsx", index=False)

    print("Saving figures...")
    save_categorical_charts(merged_df, figures_dir)
    save_numeric_boxplot(merged_df, figures_dir)

    print("Saving analysis tables...")
    save_analysis_tables(merged_df, tables_dir)

    print("Running event-study analysis...")
    run_event_study_outputs(
        merged_df,
        market_df,
        tables_dir,
        signal_only=False,
        threshold=threshold,
    )

    run_event_study_outputs(
        merged_df,
        market_df,
        tables_dir,
        signal_only=True,
        threshold=threshold,
    )

    print("Exporting dashboard dataset...")
    export_dashboard_dataset(
        merged_df,
        processed_dir / "dashboard_dataset.csv",
    )

    print("Done. Outputs saved in:")
    print(f"- Processed data: {processed_dir}")
    print(f"- Tables: {tables_dir}")
    print(f"- Figures: {figures_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Telegram crypto channels, engagement, market behavior, AAR, and Hit Ratio."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/Project2_Dataset.csv"),
        help="Path to the raw Telegram dataset. CSV or Excel is supported.",
    )

    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("."),
        help="Base project folder where outputs will be saved.",
    )

    parser.add_argument(
        "--market-start",
        type=str,
        default=None,
        help="Optional market data start date, format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--market-end",
        type=str,
        default=None,
        help="Optional market data end date, format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Hit Ratio threshold based on absolute abnormal return.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        input_path=args.input,
        output_base=args.output_base,
        market_start=args.market_start,
        market_end=args.market_end,
        threshold=args.threshold,
    )
