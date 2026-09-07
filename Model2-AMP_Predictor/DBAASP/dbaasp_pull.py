#!/usr/bin/env python3
"""
DBAASP API v4.0.1 pull script.

Built strictly from the OpenAPI spec at https://dbaasp.org/v3/api-docs
(saved locally as dbaasp_openapi.json) — no parameter names or filter
semantics are assumed beyond what that spec confirms.

WHAT THIS SCRIPT DOES:
  1. Pages through GET /peptides using only the plain string/numeric
     query params confirmed safe by the spec (kingdom.value,
     complexity.value, synthesisType.value, sequenceLength.value).
     targetSpecies.value / targetGroup.value / targetObject.value are
     typed array<object> in the spec with no documented shape, so this
     script does NOT use them — using them without knowing the real
     query encoding could silently return the wrong subset.
  2. For each peptide ID returned, calls GET /peptides/{id} to pull the
     full targetActivities[] list (this is where per-species activity
     data — including whatever "no activity" looks like — actually
     lives; the list endpoint does not expose it).
  3. Writes two raw CSVs:
       peptides_list.csv        — one row per peptide (list endpoint)
       target_activities_raw.csv — one row per targetActivities entry
     No filtering or inactive/active labeling is applied. This script
     does NOT decide what counts as a negative — that convention isn't
     documented anywhere in the schema and can only be determined by
     reading real `activity` values and `note` text.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO:
  - Does not filter by target species/group/object (ambiguous param shape).
  - Does not classify any record as active/inactive.
  - Does not assume `activity == 0`, `activity == null`, or any specific
    value means "confirmed inactive." Check target_activities_raw.csv
    yourself — sort by `activity`, read the `note` field, and look for
    the actual convention (could be a threshold like ">256", could be
    a null, could be text-only in `note`) before we write the negative-
    class filter logic.

USAGE:
    python dbaasp_pull.py --test              # pulls 5 peptides only, sanity check
    python dbaasp_pull.py --kingdom Bacteria   # full pull, filtered by kingdom
    python dbaasp_pull.py                      # full pull, no filter (SLOW — check totalCount first)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://dbaasp.org"
LIST_ENDPOINT = f"{BASE_URL}/peptides"
DETAIL_ENDPOINT = f"{BASE_URL}/peptides/{{id}}"

PAGE_SIZE = 100          # unverified max — spec gives no documented cap; back off if the API errors
REQUEST_DELAY_SEC = 0.5  # politeness delay between calls; DBAASP publishes no rate-limit spec
MAX_RETRIES = 3


def _get_with_retry(url, params=None):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            print(f"  [warn] HTTP {resp.status_code} on {url} (attempt {attempt}/{MAX_RETRIES})", file=sys.stderr)
            if resp.status_code == 403:
                print("  [warn] 403 — this is the same error documented as blocking the earlier audit attempt.", file=sys.stderr)
                print(f"         Response body: {resp.text[:300]}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  [warn] request error: {e} (attempt {attempt}/{MAX_RETRIES})", file=sys.stderr)
        time.sleep(2 * attempt)
    return None


def fetch_peptide_list(kingdom=None, complexity=None, synthesis_type=None,
                        sequence_length=None, limit=PAGE_SIZE, test_mode=False):
    """Pages through GET /peptides. Returns list of SearchResultItemView dicts."""
    all_items = []
    offset = 0
    total_count = None

    while True:
        params = {"limit": (5 if test_mode else limit), "offset": offset}
        if kingdom:
            params["kingdom.value"] = kingdom
        if complexity:
            params["complexity.value"] = complexity
        if synthesis_type:
            params["synthesisType.value"] = synthesis_type
        if sequence_length:
            params["sequenceLength.value"] = sequence_length

        print(f"  fetching offset={offset} limit={params['limit']} ...")
        data = _get_with_retry(LIST_ENDPOINT, params=params)
        if data is None:
            print("  [error] giving up on this page after retries — stopping pagination.", file=sys.stderr)
            break

        if total_count is None:
            total_count = data.get("totalCount")
            print(f"  totalCount reported by API: {total_count}")

        page_items = data.get("data", [])
        if not page_items:
            break

        all_items.extend(page_items)

        if test_mode:
            break
        if total_count is not None and len(all_items) >= total_count:
            break

        offset += len(page_items)
        time.sleep(REQUEST_DELAY_SEC)

    return all_items


def fetch_peptide_detail(peptide_id):
    """Calls GET /peptides/{id}. Returns PeptideView dict or None."""
    url = DETAIL_ENDPOINT.format(id=peptide_id)
    return _get_with_retry(url)


def flatten_mixed(mixed):
    """MixedView -> name string, or '' if missing. No assumption about which field is populated."""
    if not mixed or not isinstance(mixed, dict):
        return ""
    return mixed.get("name") or mixed.get("description") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kingdom", default=None, help="e.g. Bacteria (untested exact accepted values — spec gives no enum)")
    ap.add_argument("--complexity", default=None, help="e.g. monomer")
    ap.add_argument("--synthesis-type", default=None)
    ap.add_argument("--sequence-length", default=None)
    ap.add_argument("--test", action="store_true", help="pull only 5 peptides, print raw response, exit")
    ap.add_argument("--resume", action="store_true", help="skip peptides already recorded in done_peptide_ids.txt and append to existing CSV")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: pulling peptide list ===")
    peptides = fetch_peptide_list(
        kingdom=args.kingdom,
        complexity=args.complexity,
        synthesis_type=args.synthesis_type,
        sequence_length=args.sequence_length,
        test_mode=args.test,
    )
    print(f"  got {len(peptides)} peptide records")

    if args.test:
        print("\n=== TEST MODE: raw first record ===")
        if peptides:
            import json
            print(json.dumps(peptides[0], indent=2))
        print("\nCheck the shape above against what you expect before running a full pull.")
        return

    if not peptides:
        print("[error] no peptides returned — check filters/connectivity before proceeding.", file=sys.stderr)
        sys.exit(1)

    list_csv = outdir / "peptides_list.csv"
    with open(list_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "dbaaspId", "name", "sequence", "sequenceLength",
            "complexity", "synthesisType", "activity", "pdb", "pubchemCid",
        ])
        writer.writeheader()
        for p in peptides:
            writer.writerow({k: p.get(k, "") for k in writer.fieldnames})
    print(f"  wrote {list_csv}")

    print("\n=== Step 2: pulling target activity detail per peptide ===")
    print(f"  ({len(peptides)} detail calls — at {REQUEST_DELAY_SEC}s/call this is roughly "
          f"{len(peptides) * REQUEST_DELAY_SEC / 3600:.1f}+ hours minimum. Resumable — safe to Ctrl+C and rerun.)")

    activity_csv = outdir / "target_activities_raw.csv"
    done_ids_file = outdir / "done_peptide_ids.txt"
    fieldnames = [
        "peptideId", "sequence", "has_lowercase", "targetSpecies", "activityMeasureGroup",
        "activityMeasureValue", "concentration", "unit", "medium",
        "activity_float", "note", "reference",
    ]

    done_ids = set()
    if args.resume and done_ids_file.exists():
        done_ids = {int(x) for x in done_ids_file.read_text().split() if x.strip()}
        print(f"  --resume: skipping {len(done_ids)} peptides already fetched")

    write_header = not (activity_csv.exists() and args.resume)
    activity_file = open(activity_csv, "a" if args.resume else "w", newline="")
    done_file = open(done_ids_file, "a" if args.resume else "w")
    writer = csv.DictWriter(activity_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    lowercase_seq_count = 0
    total_rows = 0
    try:
        for i, p in enumerate(peptides, 1):
            pid = p["id"]
            if pid in done_ids:
                continue
            if i % 25 == 0 or i == 1:
                print(f"  [{i}/{len(peptides)}] peptide id={pid}")
            detail = fetch_peptide_detail(pid)
            if detail is None:
                print(f"  [warn] failed to fetch detail for peptide {pid}, skipping (not marked done — will retry on --resume)", file=sys.stderr)
                continue

            seq = detail.get("sequence", "") or ""
            has_lower = any(c.islower() for c in seq)
            if has_lower:
                lowercase_seq_count += 1

            for ta in detail.get("targetActivities", []) or []:
                writer.writerow({
                    "peptideId": ta.get("peptideId", pid),
                    "sequence": seq,
                    "has_lowercase": has_lower,
                    "targetSpecies": flatten_mixed(ta.get("targetSpecies")),
                    "activityMeasureGroup": flatten_mixed(ta.get("activityMeasureGroup")),
                    "activityMeasureValue": ta.get("activityMeasureValue", ""),
                    "concentration": ta.get("concentration", ""),
                    "unit": flatten_mixed(ta.get("unit")),
                    "medium": flatten_mixed(ta.get("medium")),
                    "activity_float": ta.get("activity", ""),
                    "note": ta.get("note", ""),
                    "reference": ta.get("reference", ""),
                })
                total_rows += 1

            done_file.write(f"{pid}\n")
            done_file.flush()
            activity_file.flush()
            time.sleep(REQUEST_DELAY_SEC)
    except KeyboardInterrupt:
        print("\n  [interrupted] progress saved. Rerun with --resume to continue from here.", file=sys.stderr)
    finally:
        activity_file.close()
        done_file.close()

    print(f"  wrote {total_rows} new rows to {activity_csv}")
    print(f"  peptides with lowercase (likely D-amino acid) residues in this run: {lowercase_seq_count}")
    print("  D-amino acids are NOT a bug — it's DBAASP's stereochemistry convention. Decide deliberately")
    print("  whether to uppercase-and-keep or exclude these before they reach ProtParam/modlAMP/ESM2.")

    print("\n=== DONE. Next step is manual, not scripted: ===")
    print("Open target_activities_raw.csv and look at:")
    print("  - the distribution of `activity_float` (sort it — is there a cluster")
    print("    at 0, or absurdly high values, that looks like a sentinel for 'inactive'?)")
    print("  - the `note` field for free text like 'no activity', 'inactive',")
    print("    '>256', 'not active up to X' — this is the likeliest place the")
    print("    inactive convention actually lives, since the schema has no")
    print("    dedicated boolean field for it.")
    print("Do NOT assume a convention and hardcode it. Confirm it against the data first.")


if __name__ == "__main__":
    main()
