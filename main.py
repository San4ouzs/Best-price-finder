
import re
import os
import csv
import json
import time
import math
import asyncio
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse, urlencode

import httpx
from bs4 import BeautifulSoup
import yaml
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process
from dotenv import load_dotenv

# --------- Config & Models ---------

load_dotenv()

DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "EUR")
EXCHANGE_API = os.getenv("EXCHANGE_API", "https://api.exchangerate.host/latest")
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "duckduckgo")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
)

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9,ru;q=0.8"}

CURRENCY_SIGNS = {
    "€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY", "₽": "RUB", "₴": "UAH", "zł": "PLN", "₺": "TRY", "C$": "CAD"
}

ATTR_RE = re.compile(r"::attr\\(([^)]+)\\)$")

class SiteConfig(BaseModel):
    domain: str
    country: Optional[str] = None
    currency: Optional[str] = None
    price_selectors: List[str] = Field(default_factory=list)
    blocklist_selectors: List[str] = Field(default_factory=list)

class Config(BaseModel):
    sites: List[SiteConfig] = Field(default_factory=list)

@dataclass
class Offer:
    title: str
    url: str
    price_raw: str
    price_value: float
    currency: str
    currency_to: Optional[str] = None
    converted_value: Optional[float] = None
    domain: Optional[str] = None

# --------- Utils ---------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""

PRICE_NUMBER_RE = re.compile(r"(?<!\\d)(\\d{1,3}([\\s.,]\\d{3})*|\\d+)([\\s.,]\\d{2})?")

def detect_currency(text: str) -> Optional[str]:
    for sign, code in CURRENCY_SIGNS.items():
        if sign in text:
            return code
    # Heuristic ISO codes
    for code in ["EUR", "USD", "GBP", "JPY", "PLN", "TRY", "CAD", "RUB", "UAH"]:
        if re.search(rf"\\b{code}\\b", text):
            return code
    return None

def parse_price(text: str) -> Optional[Tuple[float, Optional[str], str]]:
    # Returns (value, currency, raw)
    if not text:
        return None
    cur = detect_currency(text)
    m = PRICE_NUMBER_RE.search(text.replace(u"\xa0", " "))
    if not m:
        return None
    raw = m.group(0)
    # normalize decimal separator
    num = raw.replace(" ", "").replace(",", ".")
    try:
        val = float(num)
    except ValueError:
        # handle thousands separator variants like "1.299,00"
        parts = raw.replace(" ", "").split(",")
        if len(parts) == 2 and parts[1].isdigit():
            try:
                val = float(parts[0].replace(".", "") + "." + parts[1])
            except Exception:
                return None
        else:
            return None
    return (val, cur, raw)

async def fetch(url: str, client: httpx.AsyncClient, timeout=20) -> Optional[str]:
    try:
        resp = await client.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        return None
    return None

def extract_by_selectors(soup: BeautifulSoup, selectors: List[str]) -> List[str]:
    results = []
    for sel in selectors:
        attr_match = ATTR_RE.search(sel)
        attr_name = None
        css = sel
        if attr_match:
            attr_name = attr_match.group(1)
            css = sel[: attr_match.start()]
        for el in soup.select(css):
            if attr_name:
                val = el.get(attr_name)
                if val:
                    results.append(str(val))
            else:
                txt = el.get_text(" ", strip=True)
                if txt:
                    results.append(txt)
    return results

def lowest_price_from_texts(texts: List[str]) -> Optional[Tuple[float, str, str]]:
    best = None
    for t in texts:
        parsed = parse_price(t)
        if not parsed:
            continue
        val, cur, raw = parsed
        # Ignore unrealistic zeros
        if val <= 0.01:
            continue
        if not best or val < best[0]:
            best = (val, cur, raw)
    return best

async def ddg_search(query: str, client: httpx.AsyncClient, max_links=20) -> List[str]:
    q = urlencode({"q": query})
    url = f"https://duckduckgo.com/html/?{q}"
    html = await fetch(url, client)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a.result__a"):
        href = a.get("href")
        if href and href.startswith("http"):
            links.append(href)
        if len(links) >= max_links:
            break
    return links

async def get_exchange_rates(base: str) -> Dict[str, float]:
    url = f"{EXCHANGE_API}?base={base}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
            return data.get("rates", {})
    except Exception:
        return {}

async def scrape_offer(url: str, cfg: Config, fast=False) -> Optional[Offer]:
    dom = domain_of(url)
    # Match site config by domain suffix
    site = None
    for s in cfg.sites:
        if s.domain.replace("www.", "") in dom:
            site = s
            break

    async with httpx.AsyncClient(timeout=30) as client:
        html = await fetch(url, client)
        if not html:
            return None

    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else url

    texts = []
    if site and site.price_selectors:
        texts.extend(extract_by_selectors(soup, site.price_selectors))

    # Fallback: try common microdata/meta
    if not texts:
        texts.extend(extract_by_selectors(soup, [
            '[itemprop="price"]',
            'meta[itemprop="price"]::attr(content)',
            'meta[property="product:price:amount"]::attr(content)',
            '.price', '.product-price', '.Price', '[data-price]'
        ]))

    # Full-page scan fallback (heuristic; may be noisy)
    if not texts:
        texts.append(soup.get_text(" ", strip=True))

    best = lowest_price_from_texts(texts)
    if not best:
        return None
    val, cur, raw = best
    if not cur and site and site.currency:
        cur = site.currency
    if not cur:
        # last resort: infer by domain TLD (very rough)
        if dom.endswith(".de") or ".de/" in url:
            cur = "EUR"
        elif dom.endswith(".co.uk") or dom.endswith(".uk"):
            cur = "GBP"
        else:
            cur = DEFAULT_CURRENCY

    return Offer(
        title=title[:200],
        url=url,
        price_raw=raw,
        price_value=val,
        currency=cur,
        domain=dom
    )

def sort_offers(offers: List[Offer], currency: Optional[str], rates: Dict[str, float]) -> List[Offer]:
    for o in offers:
        if currency and o.currency != currency and rates:
            # Convert to desired currency
            rate = rates.get(o.currency)
            # If rates are base=target, invert
            # We fetched rates with base=target currency below
            if rate and rate > 0:
                o.converted_value = o.price_value / rate
                o.currency_to = currency
            else:
                o.converted_value = None
        elif currency and o.currency == currency:
            o.converted_value = o.price_value
            o.currency_to = currency
        else:
            o.converted_value = None
    # Sort by converted when available, else by native
    return sorted(offers, key=lambda x: x.converted_value if x.converted_value is not None else x.price_value)

async def run(query: str, only_listed=False, max_links=20, fast=False, currency: Optional[str]=None) -> List[Offer]:
    cfg = Config(**yaml.safe_load(open("sites.yaml", "r", encoding="utf-8")))
    ensure_dir(DEFAULT_OUTPUT_DIR)

    async with httpx.AsyncClient(timeout=30, headers=HEADERS, follow_redirects=True) as client:
        links = await ddg_search(query, client, max_links=max_links)

    # Filter links if only_listed=True
    if only_listed:
        allowed = [s.domain.replace("www.", "") for s in cfg.sites]
        links = [u for u in links if any(a in domain_of(u) for a in allowed)]

    # Deduplicate by domain+path
    seen = set()
    uniq_links = []
    for u in links:
        key = urlparse(u).netloc + urlparse(u).path
        if key not in seen:
            seen.add(key)
            uniq_links.append(u)

    # Scrape concurrently with bounded semaphore
    sem = asyncio.Semaphore(5)
    results: List[Offer] = []

    async def worker(u: str):
        async with sem:
            off = await scrape_offer(u, cfg, fast=fast)
            if off:
                results.append(off)

    await asyncio.gather(*(worker(u) for u in uniq_links))

    # Optionally convert currency (fetch base = target to ease division)
    rates = {}
    if results and currency:
        rates = await get_exchange_rates(base=currency)

    sorted_offers = sort_offers(results, currency, rates)
    return sorted_offers

def save_results(offers: List[Offer], query: str, outdir: str):
    ensure_dir(outdir)
    stem = re.sub(r"[^\\w\\-]+", "_", query)[:60]
    csv_path = os.path.join(outdir, f"{stem}.csv")
    json_path = os.path.join(outdir, f"{stem}.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title", "domain", "url", "price_value", "currency", "converted_value", "currency_to", "price_raw"])
        for o in offers:
            w.writerow([o.title, o.domain, o.url, o.price_value, o.currency, o.converted_value, o.currency_to, o.price_raw])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([o.__dict__ for o in offers], f, ensure_ascii=False, indent=2)

    return csv_path, json_path

def print_table(offers: List[Offer], top: Optional[int]=None):
    from math import ceil
    rows = []
    for i, o in enumerate(offers[: top or len(offers) ], start=1):
        price_show = f"{o.price_value:.2f} {o.currency}"
        if o.converted_value is not None and o.currency_to and o.currency_to != o.currency:
            price_show += f"  (~{o.converted_value:.2f} {o.currency_to})"
        rows.append([i, price_show, o.domain or "", o.title[:60], o.url])

    # Pretty print
    col_widths = [4, 22, 24, 62]
    head = ["#", "Price", "Domain", "Title"]
    print(f"{head[0]:<{col_widths[0]}} {head[1]:<{col_widths[1]}} {head[2]:<{col_widths[2]}} {head[3]:<{col_widths[3]}}")
    print("-"*(sum(col_widths)+3))
    for r in rows:
        print(f"{str(r[0]):<{col_widths[0]}} {r[1]:<{col_widths[1]}} {r[2]:<{col_widths[2]}} {r[3]:<{col_widths[3]}}")
        print(f"    {r[4]}")

def main():
    ap = argparse.ArgumentParser(description="Best Price Finder")
    ap.add_argument("query", help="Название/модель товара для поиска (например: 'iPhone 15 128GB')")
    ap.add_argument("--only-listed", action="store_true", help="Брать ссылки только из доменов, перечисленных в sites.yaml")
    ap.add_argument("--max-links", type=int, default=20, help="Максимум ссылок из поиска")
    ap.add_argument("--fast", action="store_true", help="Быстрый режим (без рендера JS)")
    ap.add_argument("--currency", type=str, default=None, help="Привести цены к валюте (например, EUR)")
    ap.add_argument("--top", type=int, default=None, help="Показывать только топ-N дешёвых")
    ap.add_argument("--save", action="store_true", help="Сохранить результаты в CSV и JSON")
    args = ap.parse_args()

    offers = asyncio.run(run(args.query, only_listed=args.only_listed, max_links=args.max_links, fast=args.fast, currency=args.currency))

    if not offers:
        print("Ничего не найдено. Попробуйте уточнить запрос или увеличить --max-links, а также расширьте sites.yaml")
        return

    print_table(offers, top=args.top)

    if args.save:
        csv_path, json_path = save_results(offers, args.query, DEFAULT_OUTPUT_DIR)
        print(f"Сохранено: {csv_path}")
        print(f"Сохранено: {json_path}")

if __name__ == "__main__":
    main()
