#!/usr/bin/env python3
"""
Pull candidate non-AMP background peptides from UniProt (reviewed/Swiss-Prot only).

Expanded exclusion list (v2) — added after finding contamination via overlap
check against amp_positive.csv: amphibian defense peptides (magainin-family),
plant-defense/cyclotide-class peptides, bacteriocins, and lantibiotics were
present in the v1 background pool despite not carrying the original
"Antimicrobial"/"Antibiotic"/"Fungicide" keywords. All new keyword strings
below were verified live against UniProt before being added (Bacteriocin:
5452 hits, Lantibiotic: 334 hits, both confirmed real and populated;
"Cyclotide" as a keyword returned 0 hits and is NOT used — the actual
cyclotide entries carry "Plant defense" instead, which is used).

NOTE: "Knottin" is deliberately NOT excluded — it's a structural fold shared
by many non-AMP toxins and protease inhibitors, not an AMP-specific term.
Excluding it would remove large amounts of legitimate negative data for no
AMP-specific reason.

This is a FRESH pull, not incremental — a different query means a different
result set, not an appendable delta. If old checkpoint/output files exist
from a different query, this script refuses to resume from them (checks the
stored query against the current one) rather than silently corrupting output.

USAGE:
    python3 pull_uniprot_background.py --test
    python3 pull_uniprot_background.py --outdir .
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
PAGE_SIZE = 500

EXCLUDE_GO_TERMS = [
    "0042742",  # defense response to bacterium
    "0050829",  # defense response to Gram-negative bacterium
    "0050830",  # defense response to Gram-positive bacterium
    "0019731",  # antibacterial humoral response
    "0061844",  # antimicrobial humoral response
    "0050832",  # defense response to fungus
]

EXCLUDE_KEYWORDS = [
    "Antimicrobial",
    "Antibiotic",
    "Fungicide",
    "Antiviral protein",
    "Amphibian defense peptide",  # added v2 — confirmed contamination source
    "Plant defense",              # added v2 — covers cyclotides (Cycloviolacin, Chassatide, etc.)
    "Bacteriocin",                # added v2 — verified live: 5452 hits
    "Lantibiotic",                # added v2 — verified live: 334 hits
]

FIELDS = "accession,id,reviewed,length,sequence,keyword,go_id,protein_name,organism_name"
RE_NEXT_LINK = re.compile(r'<(.+)>; rel="next"')


def build_query():
    parts = ["reviewed:true"]
    for go in EXCLUDE_GO_TERMS:
        parts.append(f"NOT (go:{go})")
    for kw in EXCLUDE_KEYWORDS:
        parts.append(f'NOT (keyword:"{kw}")')
    return " AND ".join(parts)


def get_next_link(headers):
    link = headers.get("Link")
    if not link:
        return None
    match = RE_NEXT_LINK.search(link)
    return match.group(1) if match else None


def make_session():
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def run_test(query):
    print("=== TEST MODE ===")
    print(f"Query:\n{query}\n")
    params = {"query": query, "fields": FIELDS, "format": "tsv", "size": 5}
    resp = requests.get(SEARCH_URL, params=params, timeout=30)
    print(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(resp.text[:1000], file=sys.stderr)
        return
    print(resp.text)
    total = resp.headers.get("x-total-results", "unknown")
    print(f"\nx-total-results header: {total}")
    print("(compare against 566484 from v1 — should be somewhat lower now)")


def run_full_pull(query, outdir):
    out_path = outdir / "uniprot_background_raw.tsv"
    checkpoint_path = outdir / "uniprot_pull_checkpoint.json"

    session = make_session()

    if checkpoint_path.exists() and out_path.exists():
        saved = json.loads(checkpoint_path.read_text())
        if saved.get("query") != query:
            print("[FATAL] Existing checkpoint is for a DIFFERENT query than the one this "
                  "script would now run. Resuming would silently corrupt the output file.\n"
                  f"  Saved query:   {saved.get('query')}\n"
                  f"  Current query: {query}\n"
                  "Delete uniprot_pull_checkpoint.json and uniprot_background_raw.tsv "
                  "(or move them aside) before rerunning.", file=sys.stderr)
            sys.exit(1)
        next_url = saved["next_url"]
        print(f"=== RESUMING (query matches saved checkpoint) ===\n{next_url}\n")
        mode = "a"
    else:
        params = {"query": query, "fields": FIELDS, "format": "tsv", "size": PAGE_SIZE}
        next_url = requests.Request("GET", SEARCH_URL, params=params).prepare().url
        print(f"=== FULL PULL (paginated, size={PAGE_SIZE}) ===\nQuery:\n{query}\n")
        mode = "w"

    total = None
    written = 0
    page_num = 0

    with open(out_path, mode, newline="") as f:
        while next_url:
            page_num += 1
            try:
                resp = session.get(next_url, timeout=60)
            except requests.RequestException as e:
                print(f"\n[error] request failed on page {page_num}: {e}\n"
                      f"Checkpoint preserved — rerun the same command to resume.", file=sys.stderr)
                sys.exit(1)

            if resp.status_code != 200:
                print(f"\n[error] HTTP {resp.status_code} on page {page_num}. "
                      f"Checkpoint preserved — rerun to resume.", file=sys.stderr)
                print(resp.text[:500], file=sys.stderr)
                sys.exit(1)

            if total is None:
                total = resp.headers.get("x-total-results", "?")

            lines = resp.text.splitlines()
            if mode == "w" and page_num == 1:
                f.write(lines[0] + "\n")
            for line in lines[1:]:
                f.write(line + "\n")
            written += len(lines) - 1
            f.flush()

            next_url = get_next_link(resp.headers)
            if next_url:
                checkpoint_path.write_text(json.dumps({"query": query, "next_url": next_url}))
            elif checkpoint_path.exists():
                checkpoint_path.unlink()

            print(f"\r  page {page_num}: {written} / {total} rows written", end="", flush=True)
            time.sleep(0.1)

    print(f"\nDone. Wrote {out_path} ({written} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    query = build_query()

    if args.test:
        run_test(query)
    else:
        run_full_pull(query, outdir)


if __name__ == "__main__":
    main()
