#!/usr/bin/env python3
"""Turn Android signing-certificate fingerprints into the WebAuthn origins the API must accept for passkeys.

  python scripts/android_origin.py "AA:BB:...:FF" ["CC:DD:...:EE" ...]
  python scripts/android_origin.py --line "AA:BB:...:FF" ["CC:DD:...:EE" ...]        (the whole server setting, usual web origins)
  python scripts/android_origin.py --web-origins https://showmefire.org,... "AA:BB:...:FF"   (your own web origins)

A passkey signed in the Android app reports its origin as `android:apk-key-hash:<base64url of the SHA-256 of the signing
certificate>`, not an https address, so each key an installed build may be signed with has to be listed in WEBAUTHN_ORIGINS:
  - the Play App Signing key (Play Console -> Setup -> App signing; this is what store installs carry),
  - the upload key (what internal/sideloaded builds made with EAS carry; `eas credentials` or `keytool -list -v`),
  - a debug/preview key if you test with one.
Give the SHA-256 fingerprint exactly as those screens show it (colon-separated hex). The same fingerprints go, unchanged, into the
website's NUXT_ANDROID_ASSET_LINKS_SHA256 (assetlinks.json).

Setting WEBAUTHN_ORIGINS REPLACES the built-in list of website origins (api/core/admin_auth/mfa.py::webauthn_settings), so
the setting must list every website origin as well, or the website's own passkeys stop working. Use --line (or --web-origins)
to print the complete WEBAUTHN_RP_ID / WEBAUTHN_ORIGINS lines.
"""
import argparse
import base64
import sys

DEFAULT_WEB = "https://showmefire.org,https://www.showmefire.org,https://preview.showmefire.org,https://dev.showmefire.org"


def fingerprint_to_origin(fingerprint: str) -> str:
    """`android:apk-key-hash:<base64url, no padding>` for a SHA-256 certificate fingerprint (hex, colons optional)."""
    cleaned = "".join(ch for ch in fingerprint.strip() if ch not in ": \\t")
    try:
        raw = bytes.fromhex(cleaned)
    except ValueError as error:
        raise ValueError(f"not a hex fingerprint: {fingerprint!r}") from error
    if len(raw) != 32:
        raise ValueError(f"expected a SHA-256 fingerprint (32 bytes), got {len(raw)} bytes: {fingerprint!r}")
    return "android:apk-key-hash:" + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fingerprints", nargs="+", help="SHA-256 certificate fingerprints, e.g. AA:BB:...:FF")
    parser.add_argument("--line", action="store_true", help="print the complete WEBAUTHN_RP_ID and WEBAUTHN_ORIGINS lines, with the usual website origins")
    parser.add_argument("--web-origins", help="comma-separated website origins to keep (implies --line)")
    args = parser.parse_args()
    try:
        origins = [fingerprint_to_origin(fp) for fp in args.fingerprints]
    except ValueError as error:
        sys.exit(str(error))
    if args.line or args.web_origins:
        web = [part.strip() for part in (args.web_origins or DEFAULT_WEB).split(",") if part.strip()]
        print("WEBAUTHN_RP_ID=showmefire.org")
        print("WEBAUTHN_ORIGINS=" + ",".join(web + origins))
    else:
        print("\n".join(origins))


if __name__ == "__main__":
    main()
