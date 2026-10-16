"""Skew-T plots of balloon soundings in and around Missouri (routine 00Z/12Z
and special launches), annotated with the same fire-weather metrics as the
hourly HRRR maps, the change since the previous launch, and HRRR omega.

For each site the two most recent launches and any special launch from the
last week are kept, each as its own image (sounding-<site>-YYYYMMDD-HHMMz.png),
with sounding-<site>.png always the newest. images/soundings.json lists what is
available per site (newest first, with metrics and changes) for the website.

maps.sh runs this every 15 minutes. Each site is checked for launches newer
than the last one processed (a small CSV query to the Iowa Environmental
Mesonet); new launches are archived to data/soundings/. A launch whose data is
still incomplete is retried on later runs and given up after 6 hours.
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps import station_products as spx
from services import soundings as snd
from services.cdn_thumbnails import thumbnail_name
from services.fire_weather import indices as ix
from services.fire_weather.summary import SECTORS, atomic_write_json

DATA_DIR = PROJECT_DIR / "data" / "soundings"
LEDGER_PATH = DATA_DIR / "processed.json"
LOOKBACK_HOURS = 36
GIVE_UP_HOURS = 6
ARCHIVE_SCAN_DAYS = 10
OMEGA_WAIT_HOURS = 3  # hold a new plot this long for HRRR omega before drawing without it
SOURCE = "Data Source: NWS radiosondes via the\nIowa Environmental Mesonet RAOB archive"
CW3E_SOURCE = "Data Source: University radiosonde (AR Recon\nCUSP program) via CW3E, UC San Diego"
# Metrics compared with the previous launch: (key, label, format, unit, section).
CHANGES = [
    ("mixing_height_ft", "mixing height", "{:+,.0f}", " ft", "fire_weather"),
    ("transport_speed_mph", "transport wind", "{:+.0f}", " mph", "fire_weather"),
    ("inversion_strength_f", "inversion", "{:+.0f}", "°F", "fire_weather"),
    ("pwat_in", "PW", "{:+.2f}", " in", "thermodynamics"),
]


def _manifest_path() -> Path:
    return spx.IMAGES_DIR / "soundings.json"


def _load_ledger() -> dict:
    try:
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _key(valid: datetime) -> str:
    return valid.strftime("%Y%m%d%H%M")


def _sector(degrees: float) -> str:
    return SECTORS[int(((degrees + 22.5) % 360) // 45)]


def _record_path(station: str, valid: datetime) -> Path:
    return DATA_DIR / f"{valid:%Y}" / f"{valid:%m}" / f"{station}_{_key(valid)}.json"


def image_name(station: str, valid: datetime) -> str:
    return f"sounding-{snd.site_slug(station)}-{valid:%Y%m%d-%H%M}z.png"


def latest_name(station: str) -> str:
    return f"sounding-{snd.site_slug(station)}.png"


def _short_time(valid: datetime) -> str:
    return f"{valid:%HZ %b %-d}"


# --- archive ----------------------------------------------------------------------

def archive(sounding: snd.Sounding, fw: dict, thermo: dict) -> None:
    s = sounding
    atomic_write_json(_record_path(s.station, s.valid), {
        "station": s.station, "valid_time": s.valid.isoformat(), "special": snd.launch_is_special(s.station, s.valid),
        "fire_weather": fw, "thermodynamics": thermo,
        "profile": {key: [None if not np.isfinite(v) else round(float(v), 2) for v in values]
                    for key, values in (("pressure_hpa", s.pressure_hpa), ("height_m", s.height_m),
                                        ("temp_c", s.temp_c), ("dewpoint_c", s.dewpoint_c),
                                        ("u_ms", s.u_ms), ("v_ms", s.v_ms))},
    })


def load_records(station: str, now: datetime) -> dict[datetime, dict]:
    """Archived launches for `station` from the last ARCHIVE_SCAN_DAYS days, by valid time."""
    cutoff = now - timedelta(days=ARCHIVE_SCAN_DAYS)
    records = {}
    months = {(cutoff + timedelta(days=day)).strftime("%Y/%m") for day in range(ARCHIVE_SCAN_DAYS + 2)}
    for month in months:
        for path in (DATA_DIR / month).glob(f"{station}_*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                valid = datetime.fromisoformat(record["valid_time"])
            except (OSError, ValueError, KeyError):
                continue
            if valid >= cutoff:
                records[valid] = record
    return records


def archive_new_launches(station: str, ledger: dict, now: datetime) -> int:
    done = set(ledger.get(station, []))
    # University (CW3E) files are sometimes posted days late in bulk, so look further back.
    start = now - (timedelta(days=ARCHIVE_SCAN_DAYS) if station in snd.CW3E_SITES else timedelta(hours=LOOKBACK_HOURS))
    added = 0
    for valid in [v for v in snd.list_launches(station, start, now) if _key(v) not in done]:
        started = time.time()
        sounding = snd.fetch_sounding(station, valid)
        if sounding is None:
            if now - valid > timedelta(hours=GIVE_UP_HOURS):
                done.add(_key(valid))  # never completed; stop asking
            print(f"{station} {valid:%Y-%m-%d %H%MZ}: incomplete profile, will retry")
            continue
        archive(sounding, snd.fire_weather_metrics(sounding), snd.thermodynamic_metrics(sounding))
        done.add(_key(valid))
        added += 1
        print(f"{station} {valid:%Y-%m-%d %H%MZ}{' (special)' if snd.launch_is_special(station, valid) else ''} archived "
              f"({time.time() - started:.1f}s)")
    ledger[station] = sorted(key for key in done if key >= _key(now - timedelta(days=ARCHIVE_SCAN_DAYS)))
    return added


# --- comparison with the previous launch ------------------------------------------

def changes(record: dict, previous: Optional[dict]) -> dict:
    if not previous:
        return {}
    result = {"since": previous["valid_time"]}
    for key, _label, _fmt, _unit, section in CHANGES:
        now_value, before = record[section].get(key), previous[section].get(key)
        if now_value is not None and before is not None:
            result[key] = round(now_value - before, 2)
    return result


def _change_line(record: dict, previous: Optional[dict], delta: dict) -> str:
    if not previous:
        return ""
    parts = [f"{label} {fmt.format(delta[key])}{unit}" for key, label, fmt, unit, _s in CHANGES
             if key in delta and fmt.format(delta[key]).lstrip("+-").strip("0.,") != ""]
    since = datetime.fromisoformat(previous["valid_time"])
    if not parts:
        return f"\n\nNo notable change since {_short_time(since)}"
    rows = [", ".join(parts[i:i + 2]) for i in range(0, len(parts), 2)]
    return f"\n\nChange since {_short_time(since)}:\n" + "\n".join(rows)


def source_text(station: str) -> str:
    return CW3E_SOURCE if station in snd.CW3E_SITES else SOURCE


def _description(fw: dict, thermo: dict, change_line: str, source: str = SOURCE) -> str:
    def value(number, fmt="{:,.0f}", unit=""):
        return "n/a" if number is None else fmt.format(number) + unit

    inversion = (f"{fw['inversion_strength_f']:.0f}°F, top {fw['inversion_top_ft']:,} ft"
                 if fw["inversion_strength_f"] >= 1 else "none")
    lines = [
        f"Mixing height (parcel method): {fw['mixing_height_ft']:,} ft AGL",
        f"Transport wind: {fw['transport_speed_mph']:.0f} mph from the {_sector(fw['transport_dir_deg'])}",
        f"Ventilation: {fw['ventilation_kt_ft']:,} kt-ft ({fw['ventilation_category'] or 'n/a'})",
        f"Surface inversion: {inversion}",
        f"0-1 km lapse rate: {value(fw['lapse_rate_c_km'], '{:.1f}', ' °C/km')}",
        f"Haines (low elevation): {value(fw['haines'])}   Hot-Dry-Windy: {value(fw['hdw'])}",
        f"Surface CAPE / CIN: {value(thermo.get('sbcape_jkg'))} / {value(thermo.get('sbcin_jkg'))} J/kg",
        f"LCL: {value(thermo.get('lcl_ft'), '{:,.0f}', ' ft AGL')}   PW: {value(thermo.get('pwat_in'), '{:.2f}', ' in')}",
    ]
    return source + "\n\n" + "\n".join(lines) + change_line


# --- plot ---------------------------------------------------------------------------

def _barb_index(s: snd.Sounding) -> np.ndarray:
    """One barb per 25 hPa below 700 hPa (the fire-weather layers) and per 50 hPa
    above, using the nearest reported wind, however many levels the sounding has."""
    winds = np.flatnonzero(np.isfinite(s.u_ms) & (s.pressure_hpa >= 100))
    chosen = []
    for target in np.concatenate([np.arange(1050, 700, -25), np.arange(700, 99, -50)]):
        if winds.size:
            nearest = winds[np.argmin(np.abs(s.pressure_hpa[winds] - target))]
            if abs(s.pressure_hpa[nearest] - target) <= (12.5 if target > 700 else 25) and nearest not in chosen:
                chosen.append(nearest)
    return np.array(chosen, dtype=int)


def _omega_panel(fig, skew_ax, omega: dict) -> None:
    """HRRR omega at the site: negative (rising air) left in blue, positive (sinking) right in red.
    Placed beside the barbs and aligned to the skew-T's drawn box so pressures line up."""
    skew_ax.apply_aspect()
    box = skew_ax.get_position()
    ax = fig.add_axes((box.x1 + 0.055, box.y0, 0.065, box.height))
    pressure = np.array(omega["pressure_hpa"], dtype=float)
    microbar = np.array(omega["omega_pa_s"], dtype=float) * 10.0  # 1 Pa/s = 10 microbar/s
    ax.set_yscale("log")
    ax.set_ylim(1050, 100)
    limit = max(5.0, float(np.nanmax(np.abs(microbar))) * 1.1)
    ax.set_xlim(-limit, limit)
    ax.fill_betweenx(pressure, 0, microbar, where=microbar < 0, color="#2b8cbe", alpha=0.8, step="mid", linewidth=0)
    ax.fill_betweenx(pressure, 0, microbar, where=microbar > 0, color="#e34a33", alpha=0.8, step="mid", linewidth=0)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_facecolor("white")
    ax.set_xlabel("Omega (µb/s)\n← rising  sinking →", fontsize=8)
    ax.set_title(omega.get("source", "HRRR").replace("HRRR ", "HRRR\n"), fontsize=8)


def render(record: dict, previous: Optional[dict], out: Path) -> None:
    import metpy.calc as mpcalc
    from metpy.plots import SkewT
    from metpy.units import units

    s = snd.sounding_from_record(record)
    fw, thermo = record["fire_weather"], record["thermodynamics"]
    name = snd.STATIONS[s.station][0]
    sp._ensure_fonts_loaded()
    fig = plt.figure(figsize=(sp.PIXEL_WIDTH / sp.MAP_DPI, sp.PIXEL_HEIGHT / sp.MAP_DPI), dpi=sp.MAP_DPI,
                     facecolor=sp.MAP_BACKGROUND)
    skew = SkewT(fig, rotation=45, rect=(0.06, 0.07, 0.42, 0.81))
    skew.ax.set_facecolor("white")
    p = s.pressure_hpa * units.hPa
    skew.plot_dry_adiabats(alpha=0.25, linewidth=0.7)
    skew.plot_moist_adiabats(alpha=0.25, linewidth=0.7)
    skew.plot_mixing_lines(alpha=0.25, linewidth=0.7)

    if previous:  # the launch before this one, faded, for at-a-glance change
        before = snd.sounding_from_record(previous)
        label = f"Previous ({_short_time(before.valid)})"
        skew.plot(before.pressure_hpa * units.hPa, before.temp_c * units.degC, color="#d62728", alpha=0.4,
                  linewidth=1.4, linestyle="--", label=label)
        ok = np.isfinite(before.dewpoint_c)
        skew.plot(before.pressure_hpa[ok] * units.hPa, before.dewpoint_c[ok] * units.degC, color="#2ca02c",
                  alpha=0.4, linewidth=1.4, linestyle="--")

    skew.plot(p, s.temp_c * units.degC, color="#d62728", linewidth=2.2, label="Temperature")
    dew = np.isfinite(s.dewpoint_c)
    skew.plot(p[dew], s.dewpoint_c[dew] * units.degC, color="#2ca02c", linewidth=2.2, label="Dewpoint")
    try:
        parcel = mpcalc.parcel_profile(p[dew], s.temp_c[0] * units.degC, s.dewpoint_c[0] * units.degC).to("degC")
        skew.plot(p[dew], parcel, color="black", linewidth=1.2, linestyle="--", label="Surface parcel")
        skew.shade_cape(p[dew], s.temp_c[dew] * units.degC, parcel, alpha=0.15)
        skew.shade_cin(p[dew], s.temp_c[dew] * units.degC, parcel, s.dewpoint_c[dew] * units.degC, alpha=0.15)
    except Exception:  # noqa: BLE001 - the parcel is a nice-to-have on odd profiles
        pass
    barbs = _barb_index(s)
    skew.plot_barbs(p[barbs], (s.u_ms[barbs] * ix.MS_TO_KT) * units.knot,
                    (s.v_ms[barbs] * ix.MS_TO_KT) * units.knot, length=6, xloc=1.05)

    # Fire-weather markers: mixing height and the top of any surface inversion.
    if fw.get("mixing_height_hpa") and np.isfinite(fw["mixing_height_hpa"]) and fw["mixing_height_ft"] > 0:
        skew.ax.axhline(fw["mixing_height_hpa"], color="#ff7f0e", linewidth=1.6, linestyle="-.")
        skew.ax.text(0.02, fw["mixing_height_hpa"], f" Mixing height {fw['mixing_height_ft']:,} ft",
                     transform=skew.ax.get_yaxis_transform(), color="#c05800", fontsize=10, va="bottom")
    if fw["inversion_strength_f"] >= 1 and fw["inversion_top_ft"] > 0:
        top_hpa = float(np.exp(np.interp(fw["inversion_top_ft"] / ix.M_TO_FT, s.height_agl_m, np.log(s.pressure_hpa))))
        skew.ax.axhline(top_hpa, color="#1f77b4", linewidth=1.2, linestyle=":")
        skew.ax.text(0.02, top_hpa, " Surface inversion top", transform=skew.ax.get_yaxis_transform(),
                     color="#1f77b4", fontsize=10, va="top")

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 50)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    if record.get("omega"):
        _omega_panel(fig, skew.ax, record["omega"])

    local = s.valid.astimezone(spx.CENTRAL)
    kind = "Special Sounding" if record["special"] else "Sounding"
    sp.add_branding(
        fig, skew.ax, title=f"{name} {kind}",
        subtitle=f"{snd.site_slug(s.station).upper()}  {s.valid:%HZ %a %b %-d, %Y}  ({local:%-I:%M %p} CT)",
        description=_description(fw, thermo, _change_line(record, previous, changes(record, previous)),
                                 source_text(s.station)),
    )
    status = {"product": f"sounding_{snd.site_slug(s.station)}", "status": "updated", "valid_time": record["valid_time"],
              "special": record["special"], "last_update": spx.valid_time_label(), **fw, **thermo}
    sp.save_and_record_status(fig, out, spx.STATUS_DIR / f"sounding_{snd.site_slug(s.station)}.json", status)


# --- keep / render / prune ----------------------------------------------------------

def add_omega(plans: dict[str, list[datetime]], records: dict[str, dict[datetime, dict]], now: datetime) -> None:
    """Fetch HRRR omega once per launch time for every kept launch that lacks it."""
    wanted: dict[datetime, list[str]] = {}
    for station, valids in plans.items():
        for valid in valids:
            if "omega" not in records[station][valid]:
                wanted.setdefault(valid, []).append(station)
    for valid, stations in sorted(wanted.items()):
        started = time.time()
        try:
            found = snd.fetch_hrrr_omega(valid, stations)
        except Exception as exc:  # noqa: BLE001 - omega is supplemental
            print(f"HRRR omega for {valid:%Y-%m-%d %HZ} failed: {exc}")
            found = {}
        for station in stations:
            record = records[station][valid]
            if station in found:
                record["omega"] = found[station]
            elif now - valid >= timedelta(hours=OMEGA_WAIT_HOURS):
                record["omega"] = None  # tried long enough: draw without it
            else:
                continue
            atomic_write_json(_record_path(station, valid), record)
        print(f"HRRR omega for {valid:%Y-%m-%d %HZ} ({len(found)}/{len(stations)} sites, {time.time() - started:.1f}s)")


def site_plan(station: str, records: dict[datetime, dict], now: datetime) -> list[datetime]:
    return snd.kept_launches(list(records), now, station)


def prune_images(station: str, keep: set[str]) -> None:
    prefix = f"sounding-{snd.site_slug(station)}-"
    for path in spx.IMAGES_DIR.glob(f"{prefix}*z.png"):
        if path.name not in keep:
            path.unlink(missing_ok=True)


def sync_latest(station: str, valids: list[datetime]) -> None:
    """Keep sounding-<site>.png a copy of the site's newest drawn launch, so it
    recovers on its own if it is ever missing or out of date."""
    newest = next((spx.IMAGES_DIR / image_name(station, v) for v in valids
                   if (spx.IMAGES_DIR / image_name(station, v)).is_file()), None)
    if newest is None:
        return
    latest = spx.IMAGES_DIR / latest_name(station)
    if not latest.is_file() or latest.stat().st_mtime < newest.stat().st_mtime \
            or latest.stat().st_size != newest.stat().st_size:
        shutil.copyfile(newest, latest)
        print(f"Updated {latest.name} from {newest.name}")


def manifest_entry(station: str, record: dict, previous: Optional[dict]) -> dict:
    valid = datetime.fromisoformat(record["valid_time"])
    image = image_name(station, valid)
    return {
        "valid_time": valid.strftime("%Y-%m-%dT%H:%MZ"),
        "label": f"{valid:%HZ %a %b %-d}" + (" (Special)" if record["special"] else ""),
        "special": record["special"],
        "image": image,
        "thumbnail": thumbnail_name(image),
        "fire_weather": record["fire_weather"],
        "thermodynamics": record["thermodynamics"],
        "omega_source": (record.get("omega") or {}).get("source"),
        "change_since_previous": changes(record, previous),
    }


def build_manifest(entries: dict[str, list[dict]]) -> dict:
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "retention": {"latest_launches": snd.KEEP_LATEST, "special_days": snd.KEEP_SPECIAL_DAYS},
        "stations": {
            station: {
                "name": snd.STATIONS[station][0], "latitude": snd.STATIONS[station][1],
                "longitude": snd.STATIONS[station][2], "latest_image": latest_name(station),
                "source": "CW3E university radiosonde share" if station in snd.CW3E_SITES else "IEM RAOB archive (NWS)",
                "soundings": entries.get(station, []),
            }
            for station in snd.STATIONS
        },
    }


def write_manifest(manifest: dict) -> bool:
    """Write images/soundings.json; True when its content (ignoring the timestamp) changed."""
    path = _manifest_path()
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
        if {**old, "generated": None} == {**manifest, "generated": None}:
            return False
    except (OSError, ValueError):
        pass
    atomic_write_json(path, manifest)
    return True


def upload_manifest() -> None:
    if os.getenv("uploadForecast", "true").strip().lower() in ("false", "0", "no", "off"):
        return
    try:
        from scripts.upload_cdn import upload_to_cdn

        upload_to_cdn([_manifest_path()], ["latest/soundings.json"], ["application/json"], ["max-age=120"])
    except Exception as exc:  # noqa: BLE001 - next change uploads it again
        print(f"soundings.json upload failed: {exc}")


def main() -> None:
    now = datetime.now(timezone.utc)
    ledger = _load_ledger()
    for station in snd.STATIONS:
        try:
            archive_new_launches(station, ledger, now)
        except Exception as exc:  # noqa: BLE001 - one site (or IEM hiccup) must not block the rest
            print(f"{station}: check failed ({type(exc).__name__}: {exc}); will retry next cycle")
    atomic_write_json(LEDGER_PATH, ledger)

    records = {station: load_records(station, now) for station in snd.STATIONS}
    plans = {station: site_plan(station, records[station], now) for station in snd.STATIONS}
    missing = {station: [v for v in valids if not (spx.IMAGES_DIR / image_name(station, v)).is_file()]
               for station, valids in plans.items()}
    add_omega(missing, records, now)

    products, pending_entries = [], {}
    for station, valids in plans.items():
        history = sorted(records[station])
        pending_entries[station] = []
        for valid in valids:
            record = records[station][valid]
            older = [v for v in history if v < valid]
            previous = records[station][older[-1]] if older else None
            pending_entries[station].append((valid, manifest_entry(station, record, previous)))
            out = spx.IMAGES_DIR / image_name(station, valid)
            if valid in missing[station] and "omega" in record:

                def build(record=record, previous=previous, out=out):
                    render(record, previous, out)
                    print(f"Wrote {out.name}")

                products.append((f"{station} {valid:%Y-%m-%d %H%MZ}", build))
        prune_images(station, {image_name(station, v) for v in valids})

    failed = None
    if products:
        try:
            spx.run_products(products, label="Sounding plots")
        except SystemExit as exit_:  # a plot failed; still publish what exists
            failed = exit_
    else:
        print("Soundings: nothing new to draw.")
    for station, valids in plans.items():
        sync_latest(station, valids)
    # The manifest lists only images that actually exist (a plot still waiting
    # for omega, or one that failed, appears once it is drawn).
    entries = {station: [entry for valid, entry in items if (spx.IMAGES_DIR / image_name(station, valid)).is_file()]
               for station, items in pending_entries.items()}
    if write_manifest(build_manifest(entries)):
        upload_manifest()
    try:  # balloon vs HRRR pairs for the verification endpoint; never blocks the plots
        from services.fire_weather import verification

        paired = verification.verify_new(snd.STATIONS, now)
        if paired:
            print(f"Verification: {paired} new balloon/HRRR pairs")
    except Exception as exc:  # noqa: BLE001
        print(f"Verification step failed: {exc}")
    if failed is not None:
        raise failed


if __name__ == "__main__":
    main()
