import argparse
import concurrent.futures
import itertools
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Generator, List, Optional, Set, Tuple

import dataset
import pytz
import requests

tz = pytz.timezone("Europe/Berlin")
now = tz.localize(datetime.now())
yyyymmddHHMM = int(now.strftime("%Y%m%d%H%M"))

# Configuration
ACCESS_TOKEN = os.environ.get("KARLS_API_TOKEN")
if not ACCESS_TOKEN:
    raise RuntimeError("KARLS_API_TOKEN environment variable not set")
API_URL = "https://pep.karls.de/vk/api/erdbeerfinder/v1/get-nearest-kiosks"
GEO_JSON_FILE = "dist/karls.geo.json"
GEO_JSON_FILE_ONLY_OPEN = "dist/karls_open.geo.json"
GEO_JSON_FILE_HISTORY = "dist/karls_history.geo.json"
NUM_WORKERS = 25  # Number of parallel workers
HISTORY_WINDOW_DAYS = 14
OPEN_ENDED_V2 = 999999999999
MAX_OPEN_INTERVAL = timedelta(days=1)

# Germany bounding box
GERMANY_BOUNDS = {
    "lat_min": 47.27,  # Southwest corner (near Basel/France border)
    "lng_min": 5.87,
    "lat_max": 55.06,  # Northeast corner (near Flensburg/Poland border)
    "lng_max": 15.04,
    "lat_step": 0.09 * 2,  # 0.09 ~10 km
    "lng_step": 0.14 * 2,  # 0.14 ~10 km
}


@dataclass
class OpeningHour:
    label: str
    startTime: str
    endTime: str
    comment: Optional[str]


@dataclass
class Kiosk:
    kioskNumber: str
    kioskId: int
    kioskName: str
    locationGroup: str
    geoLat: float
    geoLng: float
    city: str
    street: str
    zipCode: str
    isOpened: bool
    distanceFromPoi: Optional[int] = None
    openingHours: Optional[List[OpeningHour]] = None
    lastSeen: Optional[str] = None


def create_spinner() -> Generator[str, None, None]:
    return itertools.cycle(["🍓", "🫐", "🍒", "🍇"])


def generate_grid_locations() -> Set[Tuple[float, float]]:
    """Generate a grid of coordinates covering Germany."""
    locations = set()
    lat = GERMANY_BOUNDS["lat_min"]
    while lat <= GERMANY_BOUNDS["lat_max"]:
        lng = GERMANY_BOUNDS["lng_min"]
        while lng <= GERMANY_BOUNDS["lng_max"]:
            locations.add((round(lat, 4), round(lng, 4)))
            lng += GERMANY_BOUNDS["lng_step"]
        lat += GERMANY_BOUNDS["lat_step"]
    return locations

def get_locations_from_db() -> Set[Tuple[float, float]]:
    """Generate a set of locations from the database items."""
    db = dataset.connect("sqlite:///karls.db")
    table = db["items"]
    locations = set()
    for row in table.all():
        lat = row["geoLat"]
        lng = row["geoLng"]
        locations.add((lat,lng))
        
    print(f"Loaded {len(locations)} locations from the database.")
    return locations

def fetch_kiosks_data(location: Tuple[float, float], retries: int = 3, delay: float = 1.0) -> List[dict]:
    headers = {
        "authorization": f"Token {ACCESS_TOKEN}",
        "content-type": "application/json",
    }
    payload = {
        "lat": location[0],
        "lng": location[1],
    }
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                import time
                time.sleep(delay)
            else:
                print(f"\nError fetching data for location {location}: {e}")
    return []

def create_geojson(kiosks_dict: Dict[int, Kiosk], only_open: bool = False) -> dict:
    geojson = {"type": "FeatureCollection", "features": []}

    for kiosk_id, kiosk in kiosks_dict.items():
        if only_open and not kiosk.isOpened:
            continue
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [kiosk.geoLng, kiosk.geoLat],  # GeoJSON uses [lng, lat]
            },
            "properties": {
                "id": kiosk.kioskId,
                "name": kiosk.kioskName,
                "number": kiosk.kioskNumber,
                "locationGroup": kiosk.locationGroup,
                "address": f"{kiosk.street}, {kiosk.zipCode} {kiosk.city}",
                "isOpened": kiosk.isOpened,
                "lastSeen": kiosk.lastSeen,
            },
        }
        geojson["features"].append(feature)

    return geojson


def compact_timestamp_to_datetime(value: int) -> Optional[datetime]:
    try:
        timestamp = datetime.strptime(str(int(value)), "%Y%m%d%H%M")
    except (TypeError, ValueError):
        return None
    return tz.localize(timestamp)


def load_recent_open_periods_from_db(
        kiosk_ids: Set[int],
        now_dt: datetime,
        window_days: int = HISTORY_WINDOW_DAYS,
) -> Dict[int, List[dict]]:
    db = dataset.connect("sqlite:///karls.db")
    history_table_v2 = db["items_history_v2"]

    window_start = now_dt - timedelta(days=window_days)
    periods_by_kiosk: Dict[int, List[dict]] = {kiosk_id: [] for kiosk_id in kiosk_ids}

    for row in history_table_v2.all():
        kiosk_id = row.get("kioskId")
        if kiosk_id not in kiosk_ids:
            continue

        start_dt = compact_timestamp_to_datetime(row.get("v1"))
        if start_dt is None:
            continue

        is_open_ended = row.get("v2") == OPEN_ENDED_V2
        end_dt = now_dt if is_open_ended else compact_timestamp_to_datetime(row.get("v2"))
        if end_dt is None or end_dt < start_dt:
            continue

        # Kiosks should not be open for longer than one day.
        if (end_dt - start_dt) > MAX_OPEN_INTERVAL:
            continue

        if end_dt <= window_start or start_dt >= now_dt:
            continue

        clipped_start = max(start_dt, window_start)
        clipped_end = min(end_dt, now_dt)
        if clipped_end <= clipped_start:
            continue

        periods_by_kiosk[kiosk_id].append(
            {
                "start": clipped_start.isoformat(),
                "end": None if is_open_ended else clipped_end.isoformat(),
                "durationMinutes": int((clipped_end - clipped_start).total_seconds() // 60),
            }
        )

    for kiosk_id in periods_by_kiosk:
        periods_by_kiosk[kiosk_id].sort(key=lambda p: p["start"], reverse=True)

    return periods_by_kiosk


def create_history_geojson(
        kiosks_dict: Dict[int, Kiosk],
        periods_by_kiosk: Dict[int, List[dict]],
        window_days: int = HISTORY_WINDOW_DAYS,
) -> dict:
    geojson = {"type": "FeatureCollection", "features": []}

    for kiosk_id, kiosk in kiosks_dict.items():
        periods = periods_by_kiosk.get(kiosk_id, [])
        open_now = kiosk.isOpened or any(period["end"] is None for period in periods)
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [kiosk.geoLng, kiosk.geoLat],
            },
            "properties": {
                "id": kiosk.kioskId,
                "name": kiosk.kioskName,
                "number": kiosk.kioskNumber,
                "locationGroup": kiosk.locationGroup,
                "address": f"{kiosk.street}, {kiosk.zipCode} {kiosk.city}",
                "openNow": open_now,
                "historyWindowDays": window_days,
                "openPeriods": periods,
            },
        }
        geojson["features"].append(feature)

    return geojson

def worker_task(location: Tuple[float, float], all_kiosks: Dict[int, Kiosk],
                locations: Set[Tuple[float, float]], done_locations: Set[Tuple[float, float]],
                locations_lock: Lock, kiosks_lock: Lock) -> None:
    kiosks_data = fetch_kiosks_data(location)

    with kiosks_lock:
        for kiosk_data in kiosks_data:
            kiosk = Kiosk(**kiosk_data)
            kiosk_location = (kiosk.geoLat, kiosk.geoLng)

            with locations_lock:
                # Add kiosk location to queue if not processed yet and not already queued
                if kiosk_location not in done_locations and kiosk_location not in locations:
                    locations.add(kiosk_location)

            all_kiosks[kiosk.kioskId] = kiosk


def collect_kiosks_data(num_workers: int = NUM_WORKERS, use_db: bool = False) -> Dict[int, Kiosk]:
    if use_db:
        locations = get_locations_from_db()
    else:
        locations = generate_grid_locations()


    done_locations = set()
    all_kiosks = {}
    spinner = create_spinner()

    # Locks for thread safety
    locations_lock = Lock()
    kiosks_lock = Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        while locations:
            # Get a batch of locations to process
            batch = set()
            with locations_lock:
                while locations and len(batch) < num_workers:
                    loc = locations.pop()
                    batch.add(loc)
                    done_locations.add(loc)

            # Submit tasks for each location in the batch
            futures = []
            for location in batch:
                future = executor.submit(
                    worker_task,
                    location,
                    all_kiosks,
                    locations,
                    done_locations,
                    locations_lock,
                    kiosks_lock
                )
                futures.append(future)

            # Wait for all tasks in this batch to complete
            concurrent.futures.wait(futures)

            # Update progress
            sys.stdout.write(
                f"\r{next(spinner)} {len(all_kiosks)} Karls kiosks found, "
                f"{len(locations)} locations left to check. Current batch size: {len(batch)}"
            )
            sys.stdout.flush()

    print(f"\nFound {len(all_kiosks)} Karls kiosks.")
    return all_kiosks

def update_kiosks_in_db(kiosks: Dict[int, Kiosk]) -> None:
    db = dataset.connect("sqlite:///karls.db")
    table = db["items"]
    history_table = db["items_history"]
    history_table_v2 = db["items_history_v2"]

    # set all kiosks to closed before updating
    table.update({ "isOpened": False, "lastUpdate": now.isoformat()}, [])

    for kiosk in kiosks.values():
        # Upsert main item
        table.upsert(
            {
                "kioskId": kiosk.kioskId,
                "kioskNumber": kiosk.kioskNumber,
                "kioskName": kiosk.kioskName,
                "locationGroup": kiosk.locationGroup,
                "geoLat": kiosk.geoLat,
                "geoLng": kiosk.geoLng,
                "city": kiosk.city,
                "street": kiosk.street,
                "zipCode": kiosk.zipCode,
                "isOpened": kiosk.isOpened,
                "lastUpdate": now.isoformat(),
                "lastSeen": now.isoformat(),
            },
            ["kioskId"]
        )
        # Insert history record
        history_table.insert(
            {
                "kioskId": kiosk.kioskId,
                "seen_at": int(now.timestamp()),
            }
        )

    for item in table.all():
        del item["id"]
        del item["lastUpdate"]
        kiosk = Kiosk(**item)

        kiosk_d = history_table_v2.find_one(kioskId=kiosk.kioskId, v2=OPEN_ENDED_V2)
        if kiosk.isOpened:
            # If kiosk is open and no open record exists, insert a new open period
            if not kiosk_d:
                # First, close any stale open-ended records for this kiosk to prevent overlaps
                # This handles race conditions where a previous run may have left an open record
                for stale in history_table_v2.find(kioskId=kiosk.kioskId, v2=OPEN_ENDED_V2):
                    history_table_v2.update(
                        {"kioskId": kiosk.kioskId, "v1": stale["v1"], "v2": yyyymmddHHMM},
                        ["kioskId", "v1"],
                    )
                history_table_v2.insert(
                    {"kioskId": kiosk.kioskId, "v1": yyyymmddHHMM, "v2": OPEN_ENDED_V2},
                )
        else:
            # If kiosk is closed and an open record exists, close the period
            if kiosk_d:
                history_table_v2.update(
                    {"kioskId": kiosk.kioskId, "v1": kiosk_d["v1"], "v2": yyyymmddHHMM},
                    ["kioskId", "v1"],
                )


def load_kiosks_from_db() -> Dict[int, Kiosk]:
    db = dataset.connect("sqlite:///karls.db")
    table = db["items"]
    kiosks = {}
    for row in table.all():
        kiosk = Kiosk(
            kioskNumber=row["kioskNumber"],
            kioskId=row["kioskId"],
            kioskName=row["kioskName"],
            locationGroup=row["locationGroup"],
            geoLat=row["geoLat"],
            geoLng=row["geoLng"],
            distanceFromPoi=0,  # Not stored in DB, set to 0 or ignore in geojson
            city=row["city"],
            street=row["street"],
            zipCode=row["zipCode"],
            openingHours=[],  # Not stored in DB, set to empty
            isOpened=row["isOpened"],
            lastSeen=row.get("lastSeen", "unknown"),
        )
        kiosks[kiosk.kioskId] = kiosk
    return kiosks

def export_geojson_from_db():
    kiosks = load_kiosks_from_db()
    periods_by_kiosk = load_recent_open_periods_from_db(set(kiosks.keys()), now)

    with open(GEO_JSON_FILE, "w", encoding="utf-8") as f1:
        json.dump(create_geojson(kiosks), f1, ensure_ascii=False, sort_keys=True)

    with open(GEO_JSON_FILE_ONLY_OPEN, "w", encoding="utf-8") as f1:
        json.dump(create_geojson(kiosks, only_open=True), f1, ensure_ascii=False, sort_keys=True)

    with open(GEO_JSON_FILE_HISTORY, "w", encoding="utf-8") as f_history:
        json.dump(
            create_history_geojson(kiosks, periods_by_kiosk, HISTORY_WINDOW_DAYS),
            f_history,
            ensure_ascii=False,
            sort_keys=True,
        )

    total_kiosks = len(kiosks)
    open_kiosks = [k for k in kiosks.values() if k.isOpened]
    message = f"{now.strftime('%Y-%m-%d %H:%M:%S')} - {total_kiosks} kiosk locations, {len(open_kiosks)} currently open"
    
    print(message)
    
    with open("dist/karls.log", "a", encoding="utf-8") as logfile:
        logfile.write(message + "\n")

    with open("dist/karls.json", "w", encoding="utf-8") as f:
        json.dump({
            "open": [kiosk.kioskId for kiosk in open_kiosks],
        }, f, ensure_ascii=False, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Karls kiosk data collection")
    parser.add_argument("--use-db",
                        help="Use locations from the database instead of the grid",
                        action="store_true",
                        )
    args = parser.parse_args()

    print("Starting Karls kiosk data collection...")
    kiosks = collect_kiosks_data(NUM_WORKERS, use_db=args.use_db)
    update_kiosks_in_db(kiosks)
    export_geojson_from_db()

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")