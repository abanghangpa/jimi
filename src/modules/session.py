"""Session Awareness — trading session multipliers."""


def get_session(ts, config):
    """Return session name and multiplier based on UTC hour."""
    hour = ts.hour
    if 0 <= hour < 8:
        return 'ASIAN', config.get('SESSION_ASIAN_MULT', 0.85)
    elif 8 <= hour < 14:
        return 'EU', config.get('SESSION_EU_MULT', 1.0)
    elif 14 <= hour < 16:
        return 'US_OPEN', config.get('SESSION_US_OPEN_BOOST', 1.10)
    elif 16 <= hour < 22:
        return 'US', config.get('SESSION_US_MULT', 1.05)
    else:
        return 'LATE_US', config.get('SESSION_LATE_US_MULT', 0.90)
