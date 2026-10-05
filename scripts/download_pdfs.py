"""
PDF Downloader — scrapes the RBI Master Directions index and downloads the
regulatory PDFs this project is built around.

How it works:
  1. Fetch the RBI Master Directions listing page (one large HTML table).
  2. Parse every row into (category, date, title, pdf_url) with BeautifulSoup.
  3. For each target document below, pick the newest non-superseded row whose
     category and title match.
  4. Download the PDF (verifying the %PDF magic bytes) and save it under the
     filename the gold evaluation set expects.
  5. Write data/raw_pdfs/manifest.json recording what was matched and from where.

RBI consolidated its NBFC rules into topic-wise "Directions, 2025" in Nov 2025,
so several legacy documents now map to their current replacements (see notes
on each target).

Usage:
    python scripts/download_pdfs.py            # download missing PDFs
    python scripts/download_pdfs.py --list     # show matches, download nothing
    python scripts/download_pdfs.py --force    # re-download everything
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

# Add project root to path so `config` is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from bs4 import BeautifulSoup

from config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

LISTING_URL = "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx"

# RBI's firewall serves a JavaScript challenge page to non-browser user agents,
# so we identify as a regular desktop browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NBFC = "Non-Banking Financial Companies"
FIN_INCLUSION = "Financial Inclusion and Development"


@dataclass(frozen=True)
class Target:
    """A document to fetch: output filename + how to find it in the listing."""

    file_name: str
    category: str
    title_pattern: str  # case-insensitive regex matched against the row title
    note: str = ""


TARGETS: list[Target] = [
    Target(
        "Master_Direction_NBFC_Systemically_Important.pdf", NBFC,
        r"Prudential Norms on Capital Adequacy",
        "Legacy NBFC-SI direction; CRAR / Tier I rules now live here",
    ),
    Target(
        "Master_Direction_NBFC_Non_Systemically_Important.pdf", NBFC,
        r"Acceptance of Public Deposits",
        "Legacy NBFC-Non-SI direction; NOF / public deposit rules now live here",
    ),
    Target(
        "Scale_Based_Regulation_Framework_NBFC.pdf", NBFC,
        r"Framework for Scale Based Regulation",
    ),
    Target(
        "Concentration_Risk_Management_NBFC.pdf", NBFC,
        r"Concentration Risk Management",
        "Single-borrower / group exposure limits (formerly in NBFC-SI direction)",
    ),
    Target("Master_Direction_Housing_Finance_Companies.pdf", NBFC, r"\(Housing Finance Companies\)"),
    Target("Master_Direction_KYC.pdf", NBFC, r"Non-Banking Financial Companies\s*[–-]\s*Know Your Customer"),
    Target(
        "Master_Direction_Fair_Practices_Code_NBFC.pdf", NBFC,
        r"Responsible Business Conduct",
        "Fair Practices Code, recovery agents, grievance redressal, interest-rate transparency",
    ),
    Target(
        "Digital_Lending_Guidelines_2022.pdf", NBFC,
        r"Non-Banking Financial Companies\s*[–-]\s*Credit Facilities\)",
        "Digital lending, LSPs, KFS and cooling-off period are covered here",
    ),
    Target(
        "Master_Direction_IRAC_Norms_NBFC.pdf", NBFC,
        r"Income Recognition, Asset Classification and Provisioning",
    ),
    Target("Master_Direction_Outsourcing_NBFC.pdf", NBFC, r"Non-Banking Financial Companies\s*[–-]\s*Managing Risks in Outsourcing"),
    Target("Master_Direction_Microfinance_Loans.pdf", NBFC, r"Microfinance Institution"),
    Target("Corporate_Governance_NBFC.pdf", NBFC, r"Non-Banking Financial Companies\s*[–-]\s*Governance\)"),
    Target("Master_Direction_P2P_Lending.pdf", NBFC, r"Peer to Peer Lending Platform"),
    Target("Master_Direction_Account_Aggregator.pdf", NBFC, r"Account Aggregator"),
    Target(
        "Master_Direction_Priority_Sector_Lending.pdf", FIN_INCLUSION,
        r"Priority Sector Lending\s*[–-]\s*Targets and Classification\)\s*Directions",
    ),
    Target(
        "Master_Direction_Interest_Rate_NBFC.pdf", NBFC,
        r"Responsible Business Conduct",
        "Interest-rate rules for NBFCs were folded into Responsible Business Conduct; "
        "same source as the Fair Practices Code file, so it is not downloaded twice",
    ),
]

DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$")


@dataclass
class ListingRow:
    category: str
    date: str
    title: str
    pdf_url: str
    detail_url: str

    @property
    def parsed_date(self) -> datetime:
        try:
            return datetime.strptime(self.date, "%b %d, %Y")
        except ValueError:
            return datetime.min


def fetch_listing(client: httpx.Client) -> list[ListingRow]:
    """Download and parse the Master Directions index into rows."""
    logger.info("Fetching RBI Master Directions index...")
    resp = client.get(LISTING_URL)
    resp.raise_for_status()
    if "TSPD" in resp.text[:5000] and "tablebg" not in resp.text:
        raise RuntimeError("RBI returned a bot-challenge page instead of the listing")

    soup = BeautifulSoup(resp.text, "html.parser")
    rows: list[ListingRow] = []
    category = date = ""

    # The table alternates header rows (category or date) with document rows.
    for tr in soup.select("table.tablebg tr"):
        header = tr.find("td", class_="tableheader")
        if header:
            text = header.get_text(" ", strip=True)
            if DATE_RE.match(text):
                date = text
            else:
                category = text
            continue

        link = tr.find("a", class_="link2")
        pdf = tr.find("a", href=re.compile(r"\.pdf$", re.IGNORECASE))
        if not link or not pdf:
            continue

        rows.append(
            ListingRow(
                category=category,
                date=date,
                title=re.sub(r"\s+", " ", link.get_text(" ", strip=True)),
                pdf_url=urljoin(LISTING_URL, pdf["href"]),
                detail_url=urljoin(LISTING_URL, link.get("href", "")),
            )
        )

    logger.info("Parsed %d documents from the index", len(rows))
    return rows


def match_target(target: Target, rows: list[ListingRow]) -> ListingRow | None:
    """Return the newest non-superseded row matching the target."""
    pattern = re.compile(target.title_pattern, re.IGNORECASE)
    candidates = [
        r for r in rows
        if r.category == target.category
        and pattern.search(r.title)
        and "superseded" not in r.title.lower()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.parsed_date)


def download_pdf(client: httpx.Client, url: str, dest: Path, retries: int = 3) -> int:
    """Download a PDF with retries; returns size in bytes."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError(
                    f"response is not a PDF (content-type={resp.headers.get('content-type')})"
                )
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return len(resp.content)
        except Exception as e:  # noqa: BLE001 — retry on any network/content error
            last_error = e
            logger.warning("  attempt %d/%d failed: %s", attempt, retries, e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"giving up on {url}: {last_error}")


def download_pdfs(
    output_dir: str | Path | None = None,
    delay: float = 1.0,
    force: bool = False,
    list_only: bool = False,
) -> list[Path]:
    """
    Scrape the RBI index and download all target PDFs.

    Args:
        output_dir: Where to save PDFs. Defaults to the configured pdf_dir.
        delay: Seconds between downloads (be polite to rbi.org.in).
        force: Re-download files that already exist.
        list_only: Print matches without downloading.

    Returns:
        Paths of PDFs present in output_dir after the run.
    """
    settings = get_settings()
    output_dir = Path(output_dir or settings.pdf_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    saved: list[Path] = []
    url_to_file: dict[str, str] = {}

    with httpx.Client(follow_redirects=True, timeout=120.0, headers=HEADERS) as client:
        rows = fetch_listing(client)

        for target in TARGETS:
            row = match_target(target, rows)
            if row is None:
                logger.error("✗ No match for %s (pattern: %s)", target.file_name, target.title_pattern)
                manifest.append({"file_name": target.file_name, "status": "not_found"})
                continue

            entry = {"file_name": target.file_name, **asdict(row), "note": target.note}

            if list_only:
                logger.info("%-55s ← %s [%s]", target.file_name, row.title[:90], row.date)
                manifest.append({**entry, "status": "listed"})
                continue

            # Same source document already saved under another name: don't duplicate
            # it, since duplicate chunks would skew retrieval.
            if row.pdf_url in url_to_file:
                logger.info("↺ %s uses the same source as %s, skipping", target.file_name, url_to_file[row.pdf_url])
                manifest.append({**entry, "status": "alias", "alias_of": url_to_file[row.pdf_url]})
                continue

            dest = output_dir / target.file_name
            url_to_file[row.pdf_url] = target.file_name

            if dest.exists() and not force:
                logger.info("⏭  Already exists: %s", target.file_name)
                manifest.append({**entry, "status": "exists"})
                saved.append(dest)
                continue

            try:
                logger.info("⬇  %s ← %s", target.file_name, row.title[:90])
                size = download_pdf(client, row.pdf_url, dest)
                logger.info("   saved %d KB", size // 1024)
                manifest.append({**entry, "status": "downloaded", "size_bytes": size})
                saved.append(dest)
            except Exception as e:  # noqa: BLE001
                logger.error("✗ Failed: %s — %s", target.file_name, e)
                manifest.append({**entry, "status": "failed", "error": str(e)})

            time.sleep(delay)

    if not list_only:
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Manifest written to %s", manifest_path)

    unique_targets = len({t.title_pattern + t.category for t in TARGETS})
    logger.info("Done: %d / %d unique PDFs available in %s", len(saved), unique_targets, output_dir)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RBI regulatory PDFs")
    parser.add_argument("--list", action="store_true", help="Show matched documents without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between downloads")
    args = parser.parse_args()

    files = download_pdfs(delay=args.delay, force=args.force, list_only=args.list)
    if not args.list:
        print(f"\n{'=' * 60}\n{len(files)} PDF files in place")
        for f in files:
            print(f"  → {f}")
