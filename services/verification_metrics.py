"""Directional verification from full persisted comparisons, never UI samples."""
import math

FIELDS = {'Temperature (C)': 'temperature_c', 'Relative Humidity (%)': 'relative_humidity_pct',
          'Wind Speed (m/s)': 'wind_speed_ms', 'Fuel Moisture (%)': 'fuel_moisture_pct',
          'Fire Danger Index': 'fire_danger'}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize(pairs, categorical=False):
    # Triples are forecast, observed, weight. Matrix rows are observed classes.
    pairs = [(p, o, w) for p, o, w in pairs if finite(p) and finite(o) and finite(w) and w > 0]
    if categorical:
        pairs = [(p, o, w) for p, o, w in pairs if p in range(5) and o in range(5)]
    count = sum(w for _, _, w in pairs)
    if not count:
        return None
    over = sum(w for p, o, w in pairs if p > o)
    under = sum(w for p, o, w in pairs if p < o)
    bias = sum((p-o)*w for p, o, w in pairs) / count
    result = {'count': count, 'bias': round(bias, 4),
              'direction': 'over' if bias > 0.00005 else 'under' if bias < -0.00005 else 'balanced',
              'over_count': over, 'under_count': under, 'exact_count': count-over-under,
              'over_rate': over/count, 'under_rate': under/count, 'exact_rate': (count-over-under)/count,
              'forecast_mean': sum(p*w for p, _, w in pairs)/count,
              'observed_mean': sum(o*w for _, o, w in pairs)/count}
    if categorical:
        observed_critical = sum(w for _, o, w in pairs if o >= 3)
        predicted_critical = sum(w for p, _, w in pairs if p >= 3)
        misses = sum(w for p, o, w in pairs if o >= 3 and p < 3)
        false_alarms = sum(w for p, o, w in pairs if p >= 3 and o < 3)
        result.update(within_one_rate=sum(w for p, o, w in pairs if abs(p-o) <= 1)/count,
                      large_over_count=sum(w for p, o, w in pairs if p-o >= 2),
                      large_under_count=sum(w for p, o, w in pairs if o-p >= 2),
                      observed_critical_count=observed_critical, predicted_critical_count=predicted_critical,
                      critical_miss_count=misses, critical_miss_rate=misses/observed_critical if observed_critical else None,
                      critical_false_alarm_count=false_alarms,
                      critical_false_alarm_rate=false_alarms/predicted_critical if predicted_critical else None)
    return result


def directional_metrics(report):
    result = {}
    rows = report.get('comparison_rows') or []
    for label, key in FIELDS.items():
        pairs = [((row.get('forecast') or {}).get(key), (row.get('observed') or {}).get(key), 1)
                 for row in rows if isinstance(row, dict) and isinstance(row.get('forecast') or {}, dict) and isinstance(row.get('observed') or {}, dict)]
        source = 'comparison_rows'
        if label == 'Fire Danger Index':
            matrix = (report.get('confusion_matrix') or {}).get('matrix')
            if (isinstance(matrix, list) and len(matrix) == 5 and
                sum(sum(row) for row in matrix if isinstance(row, list) and all(finite(v) for v in row)) > 0 and
                all(isinstance(row, list) and len(row) == 5 and all(finite(v) and v >= 0 for v in row) for row in matrix)):
                pairs = [(p, o, weight) for o, row in enumerate(matrix) for p, weight in enumerate(row)]
                source = 'confusion_matrix'
        stats = summarize(pairs, categorical=label == 'Fire Danger Index')
        if stats:
            result[label] = {**stats, 'source': source}
        else:
            metric = report.get('metrics', {}).get(label, {})
            bias = metric.get('bias')
            if finite(bias) and metric.get('count', 0) > 0:
                result[label] = {'bias': bias, 'count': metric['count'], 'source': 'aggregate_bias',
                                 'direction': 'over' if bias > 0 else 'under' if bias < 0 else 'balanced'}
    return result
