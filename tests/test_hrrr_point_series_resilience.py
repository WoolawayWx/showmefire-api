"""The NetCDF point read must never raise, hang a request thread, or run two HDF5 reads at once (issue #46)."""
import threading
import time

from services.hrrr_hourly import store


def _fake_file(tmp_path, monkeypatch):
    path = tmp_path / "hrrr_hourly_20261116_13z.nc"
    path.write_bytes(b"x")
    monkeypatch.setattr(store, "nc_path", lambda cycle, root=None: path)
    return path


def test_transient_hdf_error_is_retried(tmp_path, monkeypatch):
    _fake_file(tmp_path, monkeypatch)
    calls = []

    def reader(path, cycle, lat, lon, names):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("NetCDF: HDF error")
        return {"cycle": cycle, "series": {}}

    monkeypatch.setattr(store, "_read_nc_point", reader)
    monkeypatch.setattr(store.time, "sleep", lambda s: None)
    assert store.point_series("20261116_13z", 37.0, -92.0)["cycle"] == "20261116_13z"
    assert len(calls) == 3


def test_persistent_failure_degrades_instead_of_raising(tmp_path, monkeypatch):
    _fake_file(tmp_path, monkeypatch)

    def reader(*args):
        raise KeyError("netCDF4 Dataset")

    monkeypatch.setattr(store, "_read_nc_point", reader)
    monkeypatch.setattr(store.time, "sleep", lambda s: None)
    assert store.point_series("20261116_13z", 37.0, -92.0) == {"error": "unavailable"}


def test_stuck_read_times_out(tmp_path, monkeypatch):
    _fake_file(tmp_path, monkeypatch)
    release = threading.Event()
    monkeypatch.setattr(store, "NC_READ_TIMEOUT_S", 0.2)
    monkeypatch.setattr(store, "_read_nc_point", lambda *a: release.wait(5))
    started = time.monotonic()
    try:
        assert store.point_series("20261116_13z", 37.0, -92.0) == {"error": "unavailable"}
        assert time.monotonic() - started < 2
    finally:
        release.set()


def test_real_reader_holds_the_global_lock(tmp_path, monkeypatch):
    """_read_nc_point takes _NC_LOCK, so two concurrent reads cannot enter HDF5 together."""
    import xarray as xr

    inside = []
    overlap = []

    class FakeDataset:
        def __enter__(self):
            inside.append(1)
            if len(inside) > 1:
                overlap.append(1)
            time.sleep(0.1)
            raise RuntimeError("stop after lock check")

        def __exit__(self, *exc):
            inside.pop()

    monkeypatch.setattr(xr, "open_dataset", lambda *a, **k: FakeDataset())
    errors = []

    def run():
        try:
            store._read_nc_point(tmp_path / "x.nc", "c", 37.0, -92.0, None)
        except RuntimeError:
            errors.append(1)
        finally:
            if inside:
                inside.pop()

    threads = [threading.Thread(target=run) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not overlap
    assert len(errors) == 3
