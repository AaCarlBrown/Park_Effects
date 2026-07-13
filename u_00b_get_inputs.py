r"""Stage 0b: download the public inputs the rerun needs.

Fetches into data\:
    parkcode.txt            Retrosheet park code / venue crosswalk
    gamelogs\glYYYY.txt     Retrosheet game logs, 1910-2025 (extracted from zips)
    wind\USAF-WBAN-YYYY.gz  NOAA ISD-lite hourly weather, O'Hare + Midway, 1950-2025

Run:  py u_00b_get_inputs.py                 (everything)
      py u_00b_get_inputs.py --wind          (one category: --wind --gamelogs --parkcode)
      py u_00b_get_inputs.py --dry-run       (list what would be fetched)

Already-present files are skipped, so it is safe to rerun after an
interruption. Total download is modest (game logs ~30 MB zipped, wind a few
MB); the script pauses briefly between requests to be polite to both servers.

Wind stations (both are tried for every year 1950-2025; whichever years exist
are kept -- Stage 5 decides the usable span):
    O'Hare  725300-94846
    Midway  725340-14819
"""
import argparse
import gzip
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile

from config import DATA_DIR, GAMELOG_DIR, PARKCODE, WIND_DIR

UA = {"User-Agent": "SABR-Mesoball research pipeline (contact: author)"}
PAUSE = 1.0          # seconds between requests (NCEI throttles aggressive clients)
RETRIES = 4
TIMEOUT = 30         # seconds; a stalled socket raises and retries instead of hanging

PARKCODE_URL = "https://www.retrosheet.org/parkcode.txt"
GAMELOG_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"
ISDLITE_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{station}-{year}.gz"
STATIONS = {"ohare": "725300-94846", "midway": "725340-14819"}
YEARS = range(1910, 2026)
WIND_YEARS = range(1950, 2026)


def fetch(url):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # legitimately absent (e.g., station-year)
            last = e
        except Exception as e:       # noqa: BLE001 - retry any transient error
            last = e
        print(f"      retry {attempt}/{RETRIES} after error: {last}")
        time.sleep(5 * attempt)      # back off harder when throttled
    raise RuntimeError(f"failed after {RETRIES} tries: {url} ({last})")


def get_parkcode(dry):
    if PARKCODE.exists():
        print(f"skip  {PARKCODE.name} (present)")
        return
    if dry:
        print(f"would fetch {PARKCODE_URL}")
        return
    data = fetch(PARKCODE_URL)
    if data is None or not data.startswith(b"PARKID"):
        print(f"ERROR parkcode.txt unexpected content from {PARKCODE_URL}")
        return
    PARKCODE.write_bytes(data)
    print(f"ok    {PARKCODE.name} ({len(data):,} bytes)")


def get_gamelogs(dry):
    GAMELOG_DIR.mkdir(parents=True, exist_ok=True)
    missing = [y for y in YEARS if not (GAMELOG_DIR / f"gl{y}.txt").exists()]
    print(f"game logs: {len(YEARS) - len(missing)} present, {len(missing)} to fetch")
    if dry:
        for y in missing[:5]:
            print(f"would fetch {GAMELOG_URL.format(year=y)}")
        if len(missing) > 5:
            print(f"... and {len(missing) - 5} more")
        return
    errors = []
    for y in missing:
        data = fetch(GAMELOG_URL.format(year=y))
        if data is None:
            errors.append(y)
            print(f"MISS  gl{y}.zip -> 404")
            continue
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
            names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if len(names) != 1:
                errors.append(y)
                print(f"ERROR gl{y}.zip: expected one .txt, found {names}")
                continue
            (GAMELOG_DIR / f"gl{y}.txt").write_bytes(zf.read(names[0]))
            print(f"ok    gl{y}.txt")
        except zipfile.BadZipFile:
            errors.append(y)
            print(f"ERROR gl{y}.zip: not a zip")
        time.sleep(PAUSE)
    if errors:
        print(f"game log years with problems: {errors}")


def gz_ok(data):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as g:
            g.read(512)
        return True
    except OSError:
        return False


def get_wind(dry):
    WIND_DIR.mkdir(parents=True, exist_ok=True)
    got, absent, errors = 0, [], []
    for label, station in STATIONS.items():
        wban = station.split("-")[1]
        for y in WIND_YEARS:
            # pre-1973 US station-years often file under USAF 999999
            candidates = [station, f"999999-{wban}"]
            if any((WIND_DIR / f"{c}-{y}.gz").exists() for c in candidates):
                continue
            miss = WIND_DIR / f"{station}-{y}.miss"
            if miss.exists():
                continue          # both names 404ed on a previous run
            if dry:
                print(f"would try {ISDLITE_URL.format(year=y, station=station)}"
                      f" (then 999999-{wban} fallback)")
                continue
            data, used = None, None
            for cand in candidates:
                url = ISDLITE_URL.format(year=y, station=cand)
                print(f"get   {cand}-{y}.gz ...", flush=True)
                data = fetch(url)
                if data is not None:
                    used = cand
                    break
            if data is None:
                absent.append((label, y))
                miss.touch()      # remember; delete the .miss file to re-probe
                continue
            out = WIND_DIR / f"{used}-{y}.gz"
            if not gz_ok(data):
                errors.append((label, y))
                print(f"ERROR {out.name}: bad gzip")
                continue
            out.write_bytes(data)
            got += 1
            print(f"ok    {out.name} ({len(data):,} bytes)")
            time.sleep(PAUSE)
    if dry:
        return
    print(f"\nwind summary: {got} downloaded")
    for label, station in STATIONS.items():
        wban = station.split("-")[1]
        have = sorted(int(p.name.split("-")[2].split(".")[0])
                      for p in WIND_DIR.glob(f"*-{wban}-*.gz"))
        if have:
            print(f"  {label} ({station}): {have[0]}-{have[-1]}, "
                  f"{len(have)} year-files")
        else:
            print(f"  {label} ({station}): NONE")
    if absent:
        gaps = {}
        for label, y in absent:
            gaps.setdefault(label, []).append(y)
        for label, ys in gaps.items():
            print(f"  {label} not on server for: {ys}")
    if errors:
        print(f"  bad files (retry later): {errors}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parkcode", action="store_true")
    ap.add_argument("--gamelogs", action="store_true")
    ap.add_argument("--wind", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    do_all = not (args.parkcode or args.gamelogs or args.wind)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if do_all or args.parkcode:
        get_parkcode(args.dry_run)
    if do_all or args.gamelogs:
        get_gamelogs(args.dry_run)
    if do_all or args.wind:
        get_wind(args.dry_run)
    print("done")


if __name__ == "__main__":
    main()
