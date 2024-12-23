def get_interval(date, full=False):
    return date.strftime("**%H:%M**" + (" *(%Y-%m-%d)*" if full else ""))


def seconds_to_minute_string(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"
