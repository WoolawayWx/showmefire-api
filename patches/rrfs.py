"""
Rapid Refresh Forecast System (RRFS)

The prototype feed this used to target (noaa-rrfs-pds/rrfs_a/...,
product natlev.3km) stopped updating 2026-08-12 when RRFS/REFS entered
their pre-implementation parallel phase ahead of full operational status
on 2026-10-06 (NWS Service Change Notice). Every request against it now
404s. The operational feed lives in a different bucket, without the
"rrfs_a/" prefix, and splits native-level fields out from the 2D surface
diagnostics (TMP/DPT/RH/UGRD/VGRD 2m-10m, APCP, DSWRF, WEASD, TCDC, HPBL,
SOILW/MSTAV) this pipeline actually needs - which now live in "2dfld"
rather than "natlev":
  rrfs.YYYYMMDD/HH/rrfs.tHHz.2dfld.13km.fFFF.na.grib2
Verified directly against the bucket listing and a sample .idx on
2026-09-08; NOAA may still adjust paths before the Oct 6 cutover.
"""

HELP = r"""
Herbie(date, model='rrfs', ...)

fxx     : int, forecast hour
product : {"2dfld.13km", "natlev.13km"}
domain  : ignored, always "na" (North America)

Example:
    Herbie("2026-09-08 12:00", model="rrfs", fxx=1, product="2dfld.13km")
"""


class rrfs:
    def template(self):
        self.DESCRIPTION = "Rapid Refresh Forecast System (RRFS) - operational"
        self.DETAILS = {
            "aws product description": "https://registry.opendata.aws/noaa-rrfs-ops/",
        }
        self.HELP = HELP

        self.PRODUCTS = {
            "2dfld.13km": "2D surface diagnostics, 13km grid, North America",
            "natlev.13km": "Native level, 13km grid, North America",
        }

        # Normalize product aliases - "natlev"/"nat" used to mean the only
        # (3km) product on the retired prototype feed; the operational feed's
        # surface diagnostics (what this pipeline needs) live in "2dfld".
        if self.product in ("nat", "natlev", "natlev3km", "natlev.3km", None):
            self.product = "2dfld.13km"

        base = "https://noaa-rrfs-ops-pds.s3.amazonaws.com"
        date = self.date
        fxx = self.fxx

        self.SOURCES = {
            "aws": (
                f"{base}/rrfs.{date:%Y%m%d/%H}/"
                f"rrfs.t{date:%H}z.{self.product}.f{fxx:03d}.na.grib2"
            ),
        }

        self.LOCALFILE = self.get_remoteFileName


class rrfs_old:
    def template(self):
        self.DESCRIPTION = "Rapid Refresh Forecast System (RRFS) Ensemble -- OLD FORMAT"
        self.DETAILS = {
            "aws product description": "https://registry.opendata.aws/noaa-rrfs/",
        }
        self.PRODUCTS = {
            "mean": "ensemble mean",
            "avrg": "ensemble products",
            "testbed.conus": "surface grids (one for each member)",
            "na": "native grids (one for each member)",
        }
        self.SOURCES = {
            "aws": f"https://noaa-rrfs-pds.s3.amazonaws.com/rrfs.{self.date:%Y%m%d/%H}/ensprod/rrfsce.t{self.date:%H}z.conus.{self.product}.f{self.fxx:02d}.grib2",
            "aws-mem": f"https://noaa-rrfs-pds.s3.amazonaws.com/rrfs.{self.date:%Y%m%d/%H}/mem{self.member:02d}/rrfs.t{self.date:%H}z.mem{self.member:02d}.{self.product}f{self.fxx:03d}.grib2",
        }
        self.LOCALFILE = f"mem{self.member:02d}/{self.get_remoteFileName}"
