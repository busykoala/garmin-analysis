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
