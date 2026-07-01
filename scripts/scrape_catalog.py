"""
Real SHL catalog scraper — BeautifulSoup + requests.

Run this on YOUR machine (not in a restricted sandbox) to build the real
data/catalog.json from SHL's public product catalog:

    python scripts/scrape_catalog.py

SHL's catalog lives at:
    https://www.shl.com/solutions/products/product-catalog/

The page is paginated and each product has its own detail page with a
short description + attributes (test type, level, duration, remote
testing). This script:
    1. Crawls the paginated listing pages to collect product URLs.
    2. Visits each product page and extracts name/description/attributes.
    3. Normalizes fields to match app/schemas.py::CatalogItem.
    4. Writes data/catalog.json.

NOTE: Selectors below are written defensively (multiple fallbacks) because
SHL's markup can change. If a selector stops matching, inspect the live
page HTML and update the CSS selectors marked with # SELECTOR.
Be a good citizen: this script sleeps between requests and sets a
descriptive User-Agent. Respect robots.txt and SHL's terms of use.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
REQUEST_DELAY_SECONDS = 1.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SHL-Catalog-Research-Bot/1.0; "
    "+educational-project)"
}

TEST_TYPE_KEYWORDS = {
    "coding": ["java", "python", "sql", "javascript", "c++", "c#", ".net",
               "coding", "programming", "developer simulation"],
    "cognitive": ["reasoning", "cognitive", "numerical", "verbal", "inductive",
                  "deductive", "verify"],
    "personality": ["personality", "opq", "occupational personality"],
    "situational_judgement": ["situational judgement", "sjt", "scenario"],
    "behavioral": ["behavioral", "behaviour", "graduate", "job focused"],
    "skills": ["sql", "excel", "typing", "data entry", "skills"],
}

LEVEL_KEYWORDS = {
    "entry": ["entry level", "entry-level", "junior"],
    "graduate": ["graduate"],
    "senior": ["senior", "lead", "manager", "leadership"],
    "mid": ["mid level", "mid-level", "professional"],
}


def guess_test_type(name: str, description: str) -> str:
    text = f"{name} {description}".lower()
    for test_type, keywords in TEST_TYPE_KEYWORDS.items():
        if any(k in text for k in keywords):
            return test_type
    return "skills"  # safe default


def guess_level(name: str, description: str) -> str:
    text = f"{name} {description}".lower()
    for level, keywords in LEVEL_KEYWORDS.items():
        if any(k in text for k in keywords):
            return level
    return "all_levels"


def extract_duration_minutes(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*(?:minutes|mins|min)\b", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def collect_product_links(catalog_url: str) -> list[str]:
    """Crawl paginated listing pages and collect all product detail URLs."""
    links: set[str] = set()
    page = 1
    while True:
        url = catalog_url if page == 1 else f"{catalog_url}?page={page}"
        print(f"[list] fetching page {page}: {url}")
        soup = fetch(url)

        # SELECTOR: product links usually sit inside the catalog table/grid.
        # Anchor tags whose href contains '/product-catalog/view/' are products.
        anchors = soup.select("a[href*='/product-catalog/view/']")
        if not anchors:
            print(f"[list] no product links found on page {page}, stopping.")
            break

        new_links = {urljoin(BASE_URL, a["href"]) for a in anchors}
        before = len(links)
        links.update(new_links)
        if len(links) == before:
            # No new links found -> we've looped past the last real page.
            break

        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)

        if page > 50:  # hard safety cap
            break

    print(f"[list] collected {len(links)} unique product URLs")
    return sorted(links)


def parse_product_page(url: str) -> Optional[dict]:
    try:
        soup = fetch(url)
    except requests.RequestException as exc:
        print(f"[warn] failed to fetch {url}: {exc}")
        return None

    # SELECTOR: title
    title_el = soup.select_one("h1") or soup.select_one("title")
    name = title_el.get_text(strip=True) if title_el else url.rstrip("/").split("/")[-1]

    # SELECTOR: description — SHL product pages usually have a summary
    # paragraph near the top of the main content area.
    desc_el = (
        soup.select_one(".product-description")
        or soup.select_one("main p")
        or soup.select_one("article p")
    )
    description = desc_el.get_text(strip=True) if desc_el else ""

    page_text = soup.get_text(" ", strip=True)
    duration = extract_duration_minutes(page_text)
    remote_testing = "remote testing" in page_text.lower() or "remote-testing" in page_text.lower()

    return {
        "name": name,
        "url": url,
        "test_type": guess_test_type(name, description),
        "description": description or f"{name} — SHL assessment.",
        "level": guess_level(name, description),
        "duration_minutes": duration,
        "remote_testing": remote_testing,
    }


def main() -> None:
    product_urls = collect_product_links(CATALOG_URL)
    catalog = []
    for i, url in enumerate(product_urls, start=1):
        print(f"[detail] ({i}/{len(product_urls)}) {url}")
        item = parse_product_page(url)
        if item:
            catalog.append(item)
        time.sleep(REQUEST_DELAY_SECONDS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(f"[done] wrote {len(catalog)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
