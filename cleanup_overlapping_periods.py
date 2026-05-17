#!/usr/bin/env python3
"""
One-time cleanup script to fix overlapping open periods in items_history_v2.

This script:
1. Finds all overlapping periods for each kiosk
2. Merges overlapping periods into single continuous periods
3. Removes duplicate/overlapping records

Run with --dry-run first to see what would be changed without modifying the database.
"""

import argparse
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple

import dataset
import pytz

tz = pytz.timezone("Europe/Berlin")
OPEN_ENDED_V2 = 999999999999


def compact_timestamp_to_datetime(value: int) -> datetime:
    """Convert compact timestamp (YYYYMMDDHHmm) to datetime."""
    if value == OPEN_ENDED_V2:
        return datetime.max.replace(tzinfo=pytz.UTC)
    timestamp = datetime.strptime(str(int(value)), "%Y%m%d%H%M")
    return tz.localize(timestamp)


def datetime_to_compact_timestamp(dt: datetime) -> int:
    """Convert datetime to compact timestamp (YYYYMMDDHHmm)."""
    if dt == datetime.max.replace(tzinfo=pytz.UTC):
        return OPEN_ENDED_V2
    return int(dt.strftime("%Y%m%d%H%M"))


def find_overlapping_periods(db) -> Dict[int, List[dict]]:
    """Find all kiosks with overlapping periods."""
    history_table = db["items_history_v2"]

    # Group all periods by kiosk
    periods_by_kiosk: Dict[int, List[dict]] = defaultdict(list)
    for row in history_table.all():
        periods_by_kiosk[row["kioskId"]].append({
            "id": row["id"],
            "kioskId": row["kioskId"],
            "v1": row["v1"],
            "v2": row["v2"],
        })

    # Find kiosks with overlapping periods
    kiosks_with_overlaps = {}
    for kiosk_id, periods in periods_by_kiosk.items():
        if len(periods) < 2:
            continue

        # Sort by start time
        periods.sort(key=lambda p: p["v1"])

        # Check for overlaps
        has_overlap = False
        for i in range(len(periods) - 1):
            current_end = periods[i]["v2"]
            next_start = periods[i + 1]["v1"]
            if current_end > next_start:
                has_overlap = True
                break

        if has_overlap:
            kiosks_with_overlaps[kiosk_id] = periods

    return kiosks_with_overlaps


def merge_overlapping_periods(periods: List[dict]) -> List[Tuple[int, int]]:
    """
    Merge overlapping periods into non-overlapping ones.
    Returns list of (v1, v2) tuples representing merged periods.
    """
    if not periods:
        return []

    # Sort by start time
    sorted_periods = sorted(periods, key=lambda p: p["v1"])

    merged = []
    current_start = sorted_periods[0]["v1"]
    current_end = sorted_periods[0]["v2"]

    for period in sorted_periods[1:]:
        if period["v1"] <= current_end:
            # Overlapping or adjacent - extend current period
            current_end = max(current_end, period["v2"])
        else:
            # Gap - save current and start new
            merged.append((current_start, current_end))
            current_start = period["v1"]
            current_end = period["v2"]

    # Don't forget the last period
    merged.append((current_start, current_end))

    return merged


def cleanup_overlaps(dry_run: bool = True):
    """Main cleanup function."""
    db = dataset.connect("sqlite:///karls.db")
    history_table = db["items_history_v2"]

    print("Scanning for overlapping periods...")
    overlaps = find_overlapping_periods(db)

    if not overlaps:
        print("✅ No overlapping periods found. Database is clean.")
        return

    print(f"Found {len(overlaps)} kiosks with overlapping periods:\n")

    total_deleted = 0
    total_updated = 0

    for kiosk_id, periods in overlaps.items():
        print(f"Kiosk {kiosk_id}:")
        print(f"  Original periods ({len(periods)}):")
        for p in sorted(periods, key=lambda x: x["v1"]):
            start_str = str(p["v1"])
            end_str = "open" if p["v2"] == OPEN_ENDED_V2 else str(p["v2"])
            print(f"    {start_str} -> {end_str}")

        merged = merge_overlapping_periods(periods)
        print(f"  Merged periods ({len(merged)}):")
        for v1, v2 in merged:
            end_str = "open" if v2 == OPEN_ENDED_V2 else str(v2)
            print(f"    {v1} -> {end_str}")

        if not dry_run:
            # Delete all existing periods for this kiosk
            for period in periods:
                history_table.delete(id=period["id"])
                total_deleted += 1

            # Insert merged periods
            for v1, v2 in merged:
                history_table.insert({
                    "kioskId": kiosk_id,
                    "v1": v1,
                    "v2": v2,
                })
                total_updated += 1

        print()

    if dry_run:
        print("=" * 50)
        print("🔍 DRY RUN - No changes made.")
        print(f"   Would fix {len(overlaps)} kiosks with overlapping periods.")
        print("   Run with --apply to make changes.")
    else:
        print("=" * 50)
        print(f"✅ Cleanup complete!")
        print(f"   Deleted {total_deleted} overlapping records")
        print(f"   Created {total_updated} merged records")


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup overlapping periods in items_history_v2"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default is dry-run)",
    )
    args = parser.parse_args()

    cleanup_overlaps(dry_run=not args.apply)


if __name__ == "__main__":
    main()

