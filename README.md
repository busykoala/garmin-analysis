# Garmin Data Extractor

This repo exports and dumps activity data from Garmin Connect.
It create a `daily_summary.csv` file and a `df_all.csv` file with the health
parameters as a time series.

![Hero Image](./assets/garmin-hero.png)

## Installation

`uv` is required to run the code.

```bash
uv sync
uv run main.py

# this will prompt you to enter your Garmin Connect credentials
# after this the data extraction will start and then the
# CSV files will be created
```

## Analysis (steps, HR, stress, mood)

After exporting, run the analysis pipeline to merge the Garmin signals with the `garmin_export/mood_tracker.csv` entries and generate plots:

```bash
uv run python -m garmin.analysis.report
```

Outputs are written to `analysis_output/` with PNG plots under `analysis_output/plots/` and a text summary in `analysis_output/summary.txt`.
