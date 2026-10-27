from datetime import date

import pytest

from services import firms_backfill as fb

HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight"
ROW = "38.5,-92.3,335.2,0.39,0.36,2026-01-05,1912,N,VIIRS,n,2.0NRT,290.1,4.2,D"


class _Response:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _fetcher(by_product):
    calls = []

    def fetch(url, timeout=None):
        calls.append(url)
        product = url.split("/csv/")[1].split("/")[1]
        return _Response(by_product.get(product, ""))
    fetch.calls = calls
    return fetch


def test_day_windows_cover_the_range_in_five_day_chunks():
    assert fb.day_windows(date(2026, 1, 1), date(2026, 1, 12)) == [
        (date(2026, 1, 1), 5), (date(2026, 1, 6), 5), (date(2026, 1, 11), 2)]
    assert fb.day_windows(date(2026, 1, 1), date(2026, 1, 1)) == [(date(2026, 1, 1), 1)]
    with pytest.raises(ValueError):
        fb.day_windows(date(2026, 1, 2), date(2026, 1, 1))


def test_archive_row_gets_the_nrt_id_and_keeps_band_fields():
    fetch = _fetcher({"VIIRS_SNPP_SP": f"{HEADER}\n{ROW}\n"})
    features, stats = fb.fetch_history(date(2026, 1, 5), date(2026, 1, 5), ["VIIRS_SNPP_SP"], key="K", fetch=fetch, sleep=lambda s: None)
    assert len(features) == 1
    props = features[0]["properties"]
    assert props["SOURCE_ID"].startswith("VIIRS_SNPP_NRT:")  # same id the live poller writes -> one fire_events row
    assert props["TYPENAME"] == "VIIRS_SNPP_NRT"
    assert (props["BRIGHT_MID"], props["BRIGHT_LONG"], props["DAYNIGHT"]) == (335.2, 290.1, "D")
    assert props["PIXEL_AREA"] == pytest.approx(0.39 * 0.36, abs=1e-4)
    assert "/VIIRS_SNPP_SP/" in fetch.calls[0] and fetch.calls[0].endswith("/1/2026-01-05")


def test_same_detection_from_archive_and_nrt_is_kept_once():
    body = f"{HEADER}\n{ROW}\n"
    fetch = _fetcher({"VIIRS_SNPP_SP": body, "VIIRS_SNPP_NRT": body})
    features, stats = fb.fetch_history(date(2026, 1, 5), date(2026, 1, 5), ["VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"], key="K",
                                       fetch=fetch, sleep=lambda s: None)
    assert len(features) == 1 and stats["duplicates"] == 1
    assert stats["per_product"] == {"VIIRS_SNPP_SP": 1, "VIIRS_SNPP_NRT": 0}


def test_detections_outside_missouri_are_dropped():
    far = ROW.replace("38.5,-92.3", "30.0,-92.3")
    fetch = _fetcher({"VIIRS_SNPP_SP": f"{HEADER}\n{far}\n"})
    features, _ = fb.fetch_history(date(2026, 1, 5), date(2026, 1, 5), ["VIIRS_SNPP_SP"], key="K", fetch=fetch, sleep=lambda s: None)
    assert features == []


def test_firms_plain_text_errors_are_raised_not_parsed_as_data():
    fetch = _fetcher({"VIIRS_SNPP_SP": "Invalid MAP_KEY."})
    with pytest.raises(fb.FirmsError, match="Invalid MAP_KEY"):
        fb.fetch_history(date(2026, 1, 5), date(2026, 1, 5), ["VIIRS_SNPP_SP"], key="K", fetch=fetch, sleep=lambda s: None)


def test_missing_map_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("MODIS_KEY", raising=False)
    with pytest.raises(fb.FirmsError, match="MODIS_KEY"):
        fb.fetch_history(date(2026, 1, 5), date(2026, 1, 5), ["VIIRS_SNPP_SP"], fetch=_fetcher({}))


def test_dry_run_writes_nothing(tmp_path):
    fetch = _fetcher({"VIIRS_SNPP_SP": f"{HEADER}\n{ROW}\n"})
    out = fb.backfill(date(2026, 1, 5), date(2026, 1, 5), products=["VIIRS_SNPP_SP"], dry_run=True, key="K",
                      out_dir=tmp_path, fetch=fetch, sleep=lambda s: None)
    assert out["fetched"] == 1 and out["dry_run"] and not list(tmp_path.iterdir())
