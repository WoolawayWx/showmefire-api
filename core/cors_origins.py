"""Browser origins allowed to call the API with credentials (see CORSMiddleware in main.py).

Exact origins go in `main.py`'s list (or the CORS_EXTRA_ORIGINS env var). Anything that needs a wildcard host
is a pattern here, since CORS itself has no wildcard syntax for hosts when credentials are used.
"""
import re

# Local development: any port on a loopback host (Nuxt/Vite pick another port when the default is taken).
LOOPBACK = r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?"
# Cloudflare Pages: the production project host and every preview deployment under it, e.g.
# https://3f2a1c9b.showmefire-website.pages.dev or https://my-branch.showmefire-website.pages.dev.
# Only that project's own subdomains match: `evil-showmefire-website.pages.dev` does not.
PAGES_PREVIEWS = r"https://(?:[a-z0-9-]+\.)*showmefire-website\.pages\.dev"

ORIGIN_REGEX = rf"^(?:{LOOPBACK}|{PAGES_PREVIEWS})$"

_compiled = re.compile(ORIGIN_REGEX)


def origin_allowed(origin: str) -> bool:
    return bool(_compiled.fullmatch(origin))
