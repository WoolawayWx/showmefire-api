"""
Rapid Refresh Forecast System (RRFS)

Updated to match current NOAA S3 bucket structure (2026):
  rrfs_a/rrfs.YYYYMMDD/HH/rrfs.tHHz.natlev.3km.fFFF.na.grib2
"""

HELP = r"""
Herbie(date, model='rrfs', ...)

fxx     : int, forecast hour
product : {"natlev.3km"}  -- only product currently available
domain  : ignored, always "na" (North America)

Example:
    Herbie("2026-02-08 12:00", model="rrfs", fxx=1, product="natlev.3km")
"""


class rrfs:
    def template(self):
        self.DESCRIPTION = "Rapid Refresh Forecast System (RRFS)"
        self.DETAILS = {
            "aws product description": "https://registry.opendata.aws/noaa-rrfs/",
        }
        self.HELP = HELP

        self.PRODUCTS = {
            "natlev.3km": "Native level, 3km grid, North America",
        }

        # Normalize product aliases
        if self.product in ("nat", "natlev", "natlev3km"):
            self.product = "natlev.3km"

        base = "https://noaa-rrfs-pds.s3.amazonaws.com"
        date = self.date
        fxx = self.fxx

        self.SOURCES = {
            "aws": (
                f"{base}/rrfs_a/rrfs.{date:%Y%m%d/%H}/"
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
