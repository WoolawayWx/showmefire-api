import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core import database
from services import fpa_fod_ingest as fpa


# Real sample attributes fetched live from the ArcGIS endpoint (see
# services/fpa_fod_ingest.py's module docstring for provenance).
SAMPLE_ATTRS = {
    "fod_id": 300205105, "fpa_id": "FS-6365152",
    "nwcg_reporting_agency": "FS", "nwcg_reporting_unit_name": "Mark Twain National Forest",
    "fire_name": "WRIGHT", "fire_code": "J41P",
    "discovery_date": 1447113600000, "discovery_time": "1500",
    "nwcg_cause_classification": "Human", "nwcg_general_cause": "Missing data/not specified/undetermined",
    "fire_size": 0.88, "latitude": 37.81611111, "longitude": -90.90861111,
    "state": "MO", "fips_code": "29221", "fips_name": "Washington County",
}


class CauseMappingTests(unittest.TestCase):
    def test_natural_maps_to_lightning(self):
        self.assertEqual(fpa._map_cause("Natural", "Natural"), "lightning")

    def test_debris_and_open_burning_maps_to_debris_burn(self):
        self.assertEqual(fpa._map_cause("Human", "Debris and open burning"), "debris_burn")

    def test_equipment_and_vehicle_use_maps_to_equipment(self):
        self.assertEqual(fpa._map_cause("Human", "Equipment and vehicle use"), "equipment")

    def test_arson_maps_to_incendiary(self):
        self.assertEqual(fpa._map_cause("Human", "Arson/incendiary"), "incendiary")

    def test_missing_data_maps_to_unknown(self):
        self.assertEqual(fpa._map_cause("Human", "Missing data/not specified/undetermined"), "unknown")

    def test_other_human_cause_falls_back_to_wildfire(self):
        self.assertEqual(fpa._map_cause("Human", "Smoking"), "wildfire")

    def test_unrecognized_classification_maps_to_unknown(self):
        self.assertEqual(fpa._map_cause(None, None), "unknown")


class OccurredAtDateMathTests(unittest.TestCase):
    def test_utc_midnight_date_is_read_directly_not_localized(self):
        # 1447113600000ms = 2015-11-10T00:00:00Z exactly (verified against
        # the sample's discovery_doy=314). If the date were localized to
        # Central BEFORE extracting the calendar date, this would come out
        # as 2015-11-09 - the exact bug the module docstring warns about.
        occurred_at, precision = fpa._occurred_at(1447113600000, "1500")
        self.assertTrue(occurred_at.startswith("2015-11-10"))
        self.assertEqual(precision, "minute")

    def test_discovery_time_is_localized_to_central_not_utc(self):
        # Nov 10 2015 is CST (UTC-6, DST ended Nov 1 2015): 15:00 CST -> 21:00Z.
        occurred_at, precision = fpa._occurred_at(1447113600000, "1500")
        self.assertEqual(occurred_at, "2015-11-10T21:00:00Z")

    def test_missing_discovery_time_falls_back_to_noon_with_day_precision(self):
        occurred_at, precision = fpa._occurred_at(1447113600000, None)
        self.assertEqual(precision, "day")
        self.assertTrue(occurred_at.startswith("2015-11-10"))

    def test_malformed_discovery_time_falls_back_to_day_precision(self):
        occurred_at, precision = fpa._occurred_at(1447113600000, "abcd")
        self.assertEqual(precision, "day")

    def test_missing_discovery_date_raises(self):
        with self.assertRaises(ValueError):
            fpa._occurred_at(None, "1500")

    def test_summer_date_uses_daylight_time_offset(self):
        # 2015-07-01T00:00:00Z is CDT (UTC-5): 15:00 CDT -> 20:00Z.
        summer_epoch_ms = 1435708800000
        occurred_at, _ = fpa._occurred_at(summer_epoch_ms, "1500")
        self.assertEqual(occurred_at, "2015-07-01T20:00:00Z")


class MapRecordTests(unittest.TestCase):
    def test_maps_real_sample_record(self):
        mapped = fpa.map_record(SAMPLE_ATTRS)
        self.assertEqual(mapped["source"], "official")
        self.assertEqual(mapped["external_id"], "fpa_fod:300205105")
        self.assertEqual(mapped["county_fips"], "29221")
        self.assertEqual(mapped["county_name"], "Washington")
        self.assertEqual(mapped["verification_tier"], "official_source_confirmed")
        self.assertEqual(mapped["cause_category"], "unknown")
        self.assertEqual(mapped["acres"], 0.88)
        self.assertEqual(mapped["official_source_system"], "USFS FPA-FOD")
        self.assertIn("FS-6365152", mapped["official_source_ref"])
        self.assertIn("WRIGHT", mapped["official_source_ref"])

    def test_missing_coordinates_returns_none(self):
        broken = {**SAMPLE_ATTRS, "latitude": None}
        self.assertIsNone(fpa.map_record(broken))

    def test_missing_fod_id_returns_none(self):
        broken = {**SAMPLE_ATTRS, "fod_id": None}
        self.assertIsNone(fpa.map_record(broken))

    def test_county_name_strips_county_suffix(self):
        mapped = fpa.map_record(SAMPLE_ATTRS)
        self.assertNotIn("County", mapped["county_name"])

    def test_falls_back_to_point_in_polygon_when_source_fips_is_missing(self):
        # Real gap confirmed against the live source: Forest Service-reported
        # fires often have fips_code=None/fips_name=None upstream (verified
        # against fpa_id='FS-1493158'). lat/lon (Columbia, MO / Boone County)
        # still let county_for_point backfill it.
        missing_fips = {**SAMPLE_ATTRS, "fips_code": None, "fips_name": None,
                        "latitude": 38.9517, "longitude": -92.3341}
        mapped = fpa.map_record(missing_fips)
        self.assertEqual(mapped["county_fips"], "29019")
        self.assertEqual(mapped["county_name"], "Boone")

    def test_uses_source_fips_when_present_without_calling_the_fallback(self):
        with patch("services.fpa_fod_ingest.county_for_point") as mock_lookup:
            mapped = fpa.map_record(SAMPLE_ATTRS)
        mock_lookup.assert_not_called()
        self.assertEqual(mapped["county_fips"], "29221")


def _fake_response(features):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"features": [{"attributes": f} for f in features]}
    return response


class FetchAndIngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.database_patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.database_patch.start()
        database.init_database()

    def tearDown(self):
        self.database_patch.stop()
        self.temporary.cleanup()

    def test_fetch_paginates_until_a_short_page(self):
        client = MagicMock()
        full_page = [dict(SAMPLE_ATTRS, fod_id=i) for i in range(3)]
        short_page = [dict(SAMPLE_ATTRS, fod_id=100)]
        client.get.side_effect = [_fake_response(full_page), _fake_response(short_page)]

        records = list(fpa.fetch_missouri_records(page_size=3, client=client))
        self.assertEqual(len(records), 4)
        self.assertEqual(client.get.call_count, 2)

    def test_fetch_raises_on_arcgis_error_payload(self):
        client = MagicMock()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"error": {"code": 400, "message": "bad where clause"}}
        client.get.return_value = response

        with self.assertRaises(RuntimeError):
            list(fpa.fetch_missouri_records(client=client))

    def test_ingest_writes_official_source_confirmed_events(self):
        # A single page shorter than page_size ends pagination without a
        # second request - one mocked response is correct here.
        client = MagicMock()
        client.get.return_value = _fake_response([SAMPLE_ATTRS])

        result = fpa.ingest_fpa_fod(client=client)
        self.assertEqual(result["inserted"], 1)

        events = database.list_fire_events(admin=True)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["verification_tier"], "official_source_confirmed")
        self.assertEqual(events[0]["source"], "official")
        self.assertEqual(events[0]["cause_category"], "unknown")

    def test_ingest_is_idempotent(self):
        client = MagicMock()
        client.get.return_value = _fake_response([SAMPLE_ATTRS])
        fpa.ingest_fpa_fod(client=client)
        result = fpa.ingest_fpa_fod(client=client)
        self.assertEqual(result["updated"], 1)
        self.assertEqual(len(database.list_fire_events(admin=True)), 1)

    def test_dry_run_writes_nothing_but_reports_would_process(self):
        client = MagicMock()
        client.get.return_value = _fake_response([SAMPLE_ATTRS])
        result = fpa.ingest_fpa_fod(dry_run=True, client=client)
        self.assertEqual(len(database.list_fire_events(admin=True)), 0)
        self.assertEqual(result["would_process"], 1)
        self.assertEqual(result["inserted"], 0)

    def test_official_records_are_immediately_exportable_as_primary_labels(self):
        client = MagicMock()
        client.get.return_value = _fake_response([SAMPLE_ATTRS])
        fpa.ingest_fpa_fod(client=client)

        labels = database.export_fire_labels(min_tier="official_source_confirmed")
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["source"], "official")


if __name__ == "__main__":
    unittest.main()
