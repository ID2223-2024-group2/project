from datetime import timedelta


def get_forecast_options(earliest_time):
    return [earliest_time + timedelta(hours=i) for i in range(0, 6)]


def get_forecast_labels(earliest_time, interval_start):
    interval_end = interval_start + timedelta(hours=1)
    fmt_start = interval_start.strftime("%H:%M")
    fmt_end = interval_end.strftime("%H:%M")
    hours_difference = int((interval_start - earliest_time).total_seconds() // 3600)
    return f"{fmt_start} - {fmt_end}  ({hours_difference + 1}h)"
