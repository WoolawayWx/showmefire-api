#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
import glob
import time
from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parent.parent.parent
dotenv_path = repo_root / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=str(dotenv_path))
else:
    load_dotenv()

DEFAULT_INCLUDES = [
    'analysis/images',
    'api/analysis/images',
    'api/analysis',
    'archive/EndOfDay',
    'images'
]


def collect_items(staging_dir: Path, includes):
    """Copy matching files/dirs into staging_dir preserving relative paths."""
    staging_dir = Path(staging_dir)
    for inc in includes:
        p = Path(inc)
        if p.exists():
            if p.is_dir():
                dest = staging_dir / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copytree(p, dest)
                except FileExistsError:
                    shutil.rmtree(dest)
                    shutil.copytree(p, dest)
            else:
                dest = staging_dir / p.parent
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, staging_dir / p)
        else:
            matches = glob.glob(inc, recursive=True)
            for m in matches:
                src = Path(m)
                rel = src
                dest = staging_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    try:
                        shutil.copytree(src, dest)
                    except FileExistsError:
                        shutil.rmtree(dest)
                        shutil.copytree(src, dest)
                else:
                    shutil.copy2(src, dest)


def make_zip_from_staging(staging_dir: Path, out_zip_path: Path):
    out_zip_path = Path(out_zip_path)
    base = out_zip_path.with_suffix('')
    archive_name = shutil.make_archive(str(base), 'zip', root_dir=str(staging_dir))
    if Path(archive_name).resolve() != out_zip_path.resolve():
        shutil.move(archive_name, out_zip_path)
    return out_zip_path


def upload_to_r2(zip_path: Path, bucket: str, account_id: str, access_key: str, secret_key: str, endpoint: str = None, object_name: str = None):
    try:
        import boto3
        from botocore.config import Config
    except Exception:
        print('boto3 not available; attempting to use upload_cdn as a fallback')
        try:
            from api.scripts import upload_cdn
        except Exception:
            try:
                import upload_cdn
            except Exception:
                print('Neither boto3 nor upload_cdn are available; cannot upload')
                return False

        key = object_name or Path(zip_path).name
        timestamp = time.strftime('%Y%m%d_%H%M')
        archive_key = f"{timestamp}/{key}"
        latest_key = f"latest/{key}"
        try:
            upload_cdn.upload_to_cdn([zip_path, zip_path], [archive_key, latest_key], content_types=['application/zip', 'application/zip'], cache_controls=['max-age=3600', 'max-age=300'])
            print(f'Uploaded {zip_path} via upload_cdn as {archive_key} and {latest_key}')
            return True
        except Exception as e:
            print('Fallback upload via upload_cdn failed:', e)
            return False

    if endpoint is None:
        if account_id:
            endpoint = f'https://{account_id}.r2.cloudflarestorage.com'
        else:
            endpoint = None

    try:
        s3 = boto3.client(
            service_name='s3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
    except Exception as e:
        print('Failed to initialize R2 client:', e)
        return False

    key = object_name or Path(zip_path).name
    try:
        s3.upload_file(str(zip_path), bucket, key)
        print(f'Uploaded {zip_path} to R2 bucket {bucket} as {key}')
        return True
    except Exception as e:
        print('R2 upload failed:', e)
        try:
            from api.scripts import upload_cdn
            timestamp = time.strftime('%Y%m%d_%H%M')
            archive_key = f"{timestamp}/{key}"
            latest_key = f"latest/{key}"
            upload_cdn.upload_to_cdn([zip_path, zip_path], [archive_key, latest_key], content_types=['application/zip', 'application/zip'], cache_controls=['max-age=3600', 'max-age=300'])
            print(f'Fallback: uploaded {zip_path} via upload_cdn as {archive_key} and {latest_key}')
            return True
        except Exception:
            return False


def parse_args():
    p = argparse.ArgumentParser(description='Package EndOfDay and forecast assets and optionally upload to Cloudflare R2')
    p.add_argument('--include', '-i', action='append', help='Files or directories to include (can be repeated). Defaults to a set of standard locations.', default=[]) 
    p.add_argument('--out', '-o', help='Output zip path', default=None)
    p.add_argument('--upload', action='store_true', help='Upload zip to Cloudflare R2 if env vars provided')
    p.add_argument('--r2-bucket', help='R2 bucket name (overrides CF_R2_BUCKET env)')
    p.add_argument('--r2-account', help='R2 account id (overrides CF_R2_ACCOUNT_ID env)')
    p.add_argument('--r2-access-key', help='R2 access key id (overrides CF_R2_ACCESS_KEY_ID env)')
    p.add_argument('--r2-secret-key', help='R2 secret access key (overrides CF_R2_SECRET_ACCESS_KEY env)')
    p.add_argument('--r2-endpoint', help='R2 endpoint URL (optional)')
    p.add_argument('--object-name', help='Remote object name to use when uploading')
    p.add_argument('--cdn', action='store_true', help='Also push the created zip to CDN via upload_cdn.upload_to_cdn')
    p.add_argument('--cdn-object-name', help='Object name to use when pushing to CDN (overrides default timestamped key)')
    return p.parse_args()


def main():
    args = parse_args()
    includes = args.include or DEFAULT_INCLUDES
    includes = [str(x) for x in dict.fromkeys(includes)]

    out_zip = args.out
    if out_zip is None:
        date_only = time.strftime('%Y%m%d')
        out_zip = f'{date_only}_archive.zip'

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix='showmefire_pkg_'))
    try:
        collect_items(staging, includes)
        zip_path = make_zip_from_staging(staging, out_zip)
        print(f'Created zip: {zip_path}')
        filename = args.object_name or Path(zip_path).name
        date_prefix = time.strftime('%Y%m%d') + '_archive'
        archive_key = f"{date_prefix}/{filename}"
        latest_key = f"latest/{filename}"
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        try:
            from api.scripts import upload_cdn
        except Exception:
            try:
                import upload_cdn
            except Exception:
                upload_cdn = None

        uploaded = False

        if upload_cdn:
            try:
                upload_cdn.upload_to_cdn([zip_path, zip_path], [archive_key, latest_key], content_types=['application/zip', 'application/zip'], cache_controls=['max-age=3600', 'max-age=300'])
                print(f'Uploaded {zip_path} via upload_cdn as {archive_key} and {latest_key}')
                uploaded = True
            except Exception as e:
                print('upload_cdn failed:', e)
                
        if not uploaded:
            bucket = args.r2_bucket or os.environ.get('R2_BUCKET') or os.environ.get('CF_R2_BUCKET')
            account = args.r2_account or os.environ.get('R2_ACCOUNT_ID') or os.environ.get('CF_R2_ACCOUNT_ID')
            access = args.r2_access_key or os.environ.get('R2_ACCESS_KEY_ID') or os.environ.get('CF_R2_ACCESS_KEY_ID')
            secret = args.r2_secret_key or os.environ.get('R2_SECRET_ACCESS_KEY') or os.environ.get('CF_R2_SECRET_ACCESS_KEY')
            endpoint = args.r2_endpoint or os.environ.get('R2_ENDPOINT') or os.environ.get('CF_R2_ENDPOINT')

            creds_present = all([bucket, account, access, secret])
            if not creds_present:
                print('Missing R2 credentials for fallback upload. Set CF_R2_BUCKET, CF_R2_ACCOUNT_ID, CF_R2_ACCESS_KEY_ID, CF_R2_SECRET_ACCESS_KEY or pass via args.')
                return 2

            ok_archive = upload_to_r2(zip_path, bucket, account, access, secret, endpoint=endpoint, object_name=archive_key)
            ok_latest = upload_to_r2(zip_path, bucket, account, access, secret, endpoint=endpoint, object_name=latest_key)
            if not (ok_archive or ok_latest):
                print('Fallback upload failed for both archive and latest keys')
                return 3
            
        try:
            zip_path.unlink()
            print(f'Deleted local zip: {zip_path}')
        except Exception as e:
            print('Warning: uploaded but failed to delete local zip:', e)
            
        if args.cdn:
            try:
                from api.scripts import upload_cdn
            except Exception:
                try:
                    import upload_cdn
                except Exception:
                    print('upload_cdn module not available; cannot push to CDN')
                    return 4

            obj_name = args.cdn_object_name or Path(zip_path).name
            timestamp = time.strftime('%Y%m%d_%H%M')
            archive_key = f"{timestamp}/{obj_name}"
            latest_key = f"latest/{obj_name}"

            try:
                upload_cdn.upload_to_cdn([zip_path, zip_path], [archive_key, latest_key], content_types=['application/zip', 'application/zip'], cache_controls=['max-age=3600', 'max-age=300'])
                print(f'Pushed zip to CDN as {archive_key} and {latest_key}')
            except Exception as e:
                print('CDN push failed:', e)
                return 5

        return 0
    finally:
        try:
            shutil.rmtree(staging)
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())
