"""One isolated hourly tick; schedule only after explicit worker enablement."""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import sys
import json
from datetime import datetime,timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.fire_weather_hybrid_job import run
from services import fire_weather_hybrid_store as store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--bundle", type=Path, help="Legacy ML bundle for comparison only; omit to run the physics path")
    parser.add_argument("--days", type=int, choices=(1,2), default=1)
    parser.add_argument("--acquire-config",type=Path,help="Acquire and condition a complete forecast instead of reading a prebuilt input")
    parser.add_argument("--allow-cold-start",action="store_true",help="Explicit staff-only initialization with class readiness flags")
    args = parser.parse_args()
    if os.getenv("SMF_FIRE_WEATHER_HYBRID_ENABLED", "false").lower() != "true":
        print("fire-weather-hybrid disabled")
        return
    source = args.input or os.getenv("SMF_FIRE_WEATHER_HYBRID_INPUT")
    bundle = args.bundle or os.getenv("SMF_FIRE_WEATHER_HYBRID_BUNDLE")
    acquisition=args.acquire_config or os.getenv("SMF_FIRE_WEATHER_HYBRID_ACQUIRE_CONFIG")
    if not (source or acquisition):
        parser.error("configure an input or acquisition JSON")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Address-space limit applies only to this isolated worker, not the API.
    if os.name != "nt":
        import resource
        limit = int(float(os.getenv("SMF_FIRE_WEATHER_HYBRID_MEMORY_GB", "8"))*1024**3)
        if limit <= 0:
            parser.error("memory limit must be positive")
        soft,hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,(min(limit,hard) if hard != resource.RLIM_INFINITY else limit,hard))
    root=store.root_dir();root.mkdir(parents=True,exist_ok=True)
    from logging.handlers import RotatingFileHandler
    logs=root/"logs";logs.mkdir(exist_ok=True)
    handler=RotatingFileHandler(logs/"pipeline.log",maxBytes=5*1024**2,backupCount=3,encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    lock=root/"pipeline.lock"
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(descriptor,str(os.getpid()).encode())
        if acquisition and not args.input:
            from services.fire_weather_hybrid_acquisition import Config,stage,install_dns_cache
            install_dns_cache()
            store.atomic_json(root/"status.json",{"state":"acquiring","updating":True,"updated_at":datetime.now(timezone.utc).isoformat()})
            source=root/"staged"/f"input-{datetime.now(timezone.utc):%Y%m%d_%H%M%S}-{os.getpid()}.nc"
            from services.fire_weather_hybrid.trees import Bundle
            stage(Config.read(acquisition),source,days=args.days,allow_cold_start=True if args.allow_cold_start else None,bundle=Bundle(bundle) if bundle else None)
        run(source,bundle or None,days=args.days)
    except Exception as error:
        logging.exception("hybrid pipeline failed")
        store.atomic_json(root/"status.json",{"state":"failed","updating":False,"error_type":type(error).__name__,
            "updated_at":datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        os.close(descriptor);lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

