import httpx
import asyncio
from datetime import datetime
from typing import Iterable, List, Set

MO_OFFICES = ["EAX", "SGF", "LSX"]
NWS_BASE = "https://api.weather.gov"
HEADERS = {"User-Agent": "your-app contact@youremail.com"}

# NWS asks you stay under ~1 req/sec
REQUEST_DELAY = 1.0  # seconds between requests

async def fetch_with_backoff(client, url, retries=3):
    """Retry with exponential backoff on failure."""
    for attempt in range(retries):
        try:
            r = await client.get(url, headers=HEADERS, timeout=10.0)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"Retrying {url} in {wait}s — {e}")
            await asyncio.sleep(wait)

async def fetch_new_afds_for_office(office: str, client: httpx.AsyncClient, known_ids: Set[str]) -> List[dict]:
    """Fetch at most one AFD: the latest product for this office if it is new."""
    results = []

    r = await fetch_with_backoff(client, f"{NWS_BASE}/products/types/AFD/locations/{office}")
    products = r.json().get("@graph", [])

    latest_product = next((product for product in products if product.get("id")), None)
    if not latest_product:
        return results

    product_id = latest_product.get("id")
    if product_id in known_ids:
        # Current latest product already stored.
        return results

    await asyncio.sleep(REQUEST_DELAY)  # rate limit between requests

    detail = await fetch_with_backoff(client, f"{NWS_BASE}/products/{product_id}")
    data = detail.json()

    results.append({
        "office": office,
        "product_id": product_id,
        "issued_at": datetime.fromisoformat(data["issuanceTime"].replace("Z", "+00:00")),
        "raw_text": data.get("productText", ""),
    })
    known_ids.add(product_id)

    return results

async def fetch_all_mo_afds(known_ids: Set[str], offices: Iterable[str] = MO_OFFICES) -> List[dict]:
    async with httpx.AsyncClient() as client:
        all_results = []
        for office in offices:
            try:
                afds = await fetch_new_afds_for_office(office, client, known_ids)
                all_results.extend(afds)
                await asyncio.sleep(REQUEST_DELAY)  # delay between offices too
            except Exception as e:
                print(f"Error fetching {office}: {e}")
        return all_results