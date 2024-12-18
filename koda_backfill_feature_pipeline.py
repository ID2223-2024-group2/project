import os
import sys
import time
import typing

import hopsworks
import pandas as pd

from koda.koda_constants import OperatorsWithRT
import koda.koda_pipeline as kp
import shared.features as sf
from shared.file_logger import setup_logger

OPERATOR = OperatorsWithRT.X_TRAFIK
RUN_HW_MATERIALIZATION_EVERY = 10

log_file_path = os.path.join(os.path.dirname(__file__), 'koda_backfill.log')
logger = setup_logger('koda_backfill', log_file_path)

# Enable copy-on-write for pandas to avoid SettingWithCopyWarning
pd.options.mode.copy_on_write = True


def backfill_date(date: str, fg=None, dry_run=True) -> (int, typing.Union[None, object]):
    df, map_df = kp.get_koda_data_for_day(date, OPERATOR)

    if df.empty:
        logger.warning(f"No data available for {date}. Pipeline exiting.")
        return 1, None

    final_metrics = sf.build_feature_group(df, map_df)

    if dry_run:
        final_metrics.to_csv("koda_backfill.csv", index=False)
        return -1, None

    if fg is None:
        logger.warning("No Hopsworks connection. Skipping upload.")
        return 2, None
    try:
        j, _ = fg.insert(final_metrics, write_options={"start_offline_materialization": False})
        sf.update_feature_descriptions(fg)
    except Exception as e:
        logger.warning(f"Failed to connect to Hopsworks and skipping upload. {e}")
        return 2, None

    return 0, j


if __name__ == "__main__":
    START_DATE = os.environ.get("START_DATE", "2024-09-07")
    END_DATE = os.environ.get("END_DATE", "2024-09-07")
    DRY_RUN = os.environ.get("DRY_RUN", "False").lower() == "true"
    STRIDE = pd.DateOffset(days=int(os.environ.get("STRIDE", 1)))

    try:
        dates = pd.date_range(START_DATE, END_DATE, freq=STRIDE)
    except (ValueError, TypeError):
        logger.error("Invalid date range. Exiting.")
        sys.exit(1)
    total_dates = len(dates)
    start_time = time.time()

    delays_fg = None
    if not DRY_RUN:
        if os.environ.get("HOPSWORKS_API_KEY") is None:
            os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()
        try:
            project = hopsworks.login()
            fs = project.get_feature_store()
            # TODO: Data expectations?
            delays_fg = fs.get_or_create_feature_group(
                name='delays',
                description='Aggregated delay metrics per hour per day',
                version=6,
                primary_key=['arrival_time_bin'],
                event_time='arrival_time_bin'
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Hopsworks. Continuing without uploads {e}")

    logger.info("Starting backfill process for dates: %s - %s and stride: %s (%s days)", START_DATE, END_DATE, STRIDE,
                total_dates)

    exit_codes = []

    for i, datetime in enumerate(dates):
        logger.info("Starting processing for date: %s", datetime)
        date = datetime.strftime("%Y-%m-%d")
        exit_code, job = backfill_date(date, fg=delays_fg, dry_run=DRY_RUN)
        if exit_code == -1:
            logger.info("Dry run completed for date: %s", date)
            sys.exit(0)
        if exit_code == 0 and job is not None and i % RUN_HW_MATERIALIZATION_EVERY == 0:
            logger.info("Running offline materialization jobs")
            try:
                job.run(await_termination=False)
            except Exception as e:
                logger.error(f"Failed to run offline materialization job. Skipping. {e}")
        exit_codes.append(exit_code)
        elapsed_time = time.time() - start_time
        avg_time_per_date = elapsed_time / (i + 1)
        remaining_time = avg_time_per_date * (total_dates - (i + 1))
        logger.info("Completed processing for date: %s with exit code: %d", date, exit_code)
        logger.info("Progress: %d/%d (%.2f%%) - Estimated time remaining: %.2f seconds",
                    i + 1, total_dates, (i + 1) / total_dates * 100, remaining_time)

    logger.info("Backfill process completed for dates: %s - %s", START_DATE, END_DATE)
    successfully_uploaded = exit_codes.count(0)
    not_uploaded = exit_codes.count(2)
    missing_data = exit_codes.count(1)
    logger.info("Summary: Successfully uploaded: %d, Not uploaded: %d, Missing data: %d", successfully_uploaded,
                not_uploaded, missing_data)
