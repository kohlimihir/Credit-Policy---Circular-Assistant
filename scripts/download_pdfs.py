"""
PDF Downloader — automated download of public RBI master directions and circulars.

Downloads PDF documents from the RBI website and organizes them
into the data/raw_pdfs/ directory with clean filenames.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import httpx
from tqdm import tqdm

from config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Known RBI Master Directions and Circulars (public PDFs) ──────────────
# These are direct PDF links from rbi.org.in — all publicly accessible.

RBI_PDF_URLS = [
    # NBFC Master Directions
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/45abormd01092016.pdf",
        "name": "Master_Direction_NBFC_Non_Systemically_Important.pdf",
        "category": "master_direction",
    },
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MDSI06092016.pdf",
        "name": "Master_Direction_NBFC_Systemically_Important.pdf",
        "category": "master_direction",
    },
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MDHFCFL5A9A930D28464F40A1CCBE35B2E29587.PDF",
        "name": "Master_Direction_Housing_Finance_Companies.pdf",
        "category": "master_direction",
    },
    # KYC Master Direction
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/57abormd25022016.pdf",
        "name": "Master_Direction_KYC.pdf",
        "category": "master_direction",
    },
    # Fair Practices Code
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/19MDFPNBFC06092016.pdf",
        "name": "Master_Direction_Fair_Practices_Code_NBFC.pdf",
        "category": "fair_practices_code",
    },
    # Digital Lending Guidelines
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/GUIDELINESNOT10082022.pdf",
        "name": "Digital_Lending_Guidelines_2022.pdf",
        "category": "guidelines",
    },
    # IRAC Norms
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/64IRACNBFC9CE23E2E5E2E4D5E8FC72F0E3FF20F8D.PDF",
        "name": "Master_Direction_IRAC_Norms_NBFC.pdf",
        "category": "master_direction",
    },
    # Outsourcing Guidelines
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MDOUTSOURCE06092016.pdf",
        "name": "Master_Direction_Outsourcing_NBFC.pdf",
        "category": "guidelines",
    },
    # Scale Based Regulation
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NT11922102021.PDF",
        "name": "Scale_Based_Regulation_Framework_NBFC.pdf",
        "category": "circular",
    },
    # Microfinance Institutions
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MFMD14032022.PDF",
        "name": "Master_Direction_Microfinance_Loans.pdf",
        "category": "master_direction",
    },
    # Corporate Governance
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NOTI3C07FA63E95D5496D99ED9C8B127D48E9.PDF",
        "name": "Corporate_Governance_NBFC.pdf",
        "category": "guidelines",
    },
    # P2P Lending Platform
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NOTIMD1906102017.pdf",
        "name": "Master_Direction_P2P_Lending.pdf",
        "category": "master_direction",
    },
    # Account Aggregator
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MDAA02092016.pdf",
        "name": "Master_Direction_Account_Aggregator.pdf",
        "category": "master_direction",
    },
    # Priority Sector Lending
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MDPSL04092020.PDF",
        "name": "Master_Direction_Priority_Sector_Lending.pdf",
        "category": "master_direction",
    },
    # Interest Rate Direction
    {
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MDIRN706092016.pdf",
        "name": "Master_Direction_Interest_Rate_NBFC.pdf",
        "category": "master_direction",
    },
]


def download_pdfs(
    output_dir: str | Path | None = None,
    urls: list[dict] | None = None,
    delay: float = 1.0,
) -> list[Path]:
    """
    Download PDF files from the RBI website.

    Args:
        output_dir: Directory to save PDFs. Defaults to configured pdf_dir.
        urls: List of URL dicts. Defaults to RBI_PDF_URLS.
        delay: Seconds to wait between downloads (respect rate limits).

    Returns:
        List of paths to successfully downloaded files.
    """
    settings = get_settings()
    output_dir = Path(output_dir or settings.pdf_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = urls or RBI_PDF_URLS
    downloaded: list[Path] = []

    logger.info("Downloading %d PDFs to %s", len(urls), output_dir)

    for entry in tqdm(urls, desc="Downloading PDFs"):
        url = entry["url"]
        name = entry["name"]
        dest = output_dir / name

        if dest.exists():
            logger.info("  ⏭️  Already exists: %s", name)
            downloaded.append(dest)
            continue

        try:
            logger.info("  ⬇️  Downloading: %s", name)
            response = httpx.get(
                url,
                follow_redirects=True,
                timeout=60.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; RBI-Policy-Assistant/1.0)",
                },
            )
            response.raise_for_status()

            # Verify it's actually a PDF
            if not response.content[:5].startswith(b"%PDF"):
                logger.warning("  ⚠️  Not a PDF: %s", name)
                continue

            dest.write_bytes(response.content)
            downloaded.append(dest)
            logger.info("  ✅ Saved: %s (%d KB)", name, len(response.content) // 1024)

            time.sleep(delay)  # rate limiting

        except Exception as e:
            logger.error("  ❌ Failed: %s — %s", name, str(e))

    logger.info(
        "Download complete: %d / %d successful", len(downloaded), len(urls)
    )
    return downloaded


if __name__ == "__main__":
    downloaded_files = download_pdfs()
    print(f"\n{'='*60}")
    print(f"Downloaded {len(downloaded_files)} PDF files")
    for f in downloaded_files:
        print(f"  → {f}")
