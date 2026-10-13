#!/usr/bin/env python3
"""
Проверка покрытия Mapillary в bbox через векторные тайлы (тот же источник, что у карты на сайте).
Запуск:  python check_area.py MIN_LON MIN_LAT MAX_LON MAX_LAT
Нужно:   pip install requests mapbox-vector-tile
Токен берётся из .env / MAPILLARY_TOKEN.
"""
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mapbox_vector_tile
import requests

TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
Z = 14  # слой отдельных снимков ("image") доступен только на зуме 14


def load_dotenv(path):
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tile_px_to_lonlat(x, y, z, px, py, extent):
    """px, py — координаты внутри тайла, ось y направлена вниз."""
    n = 2 ** z
    lon = (x + px / extent) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + py / extent) / n))))
    return lon, lat


def decode(content):
    try:  # mapbox-vector-tile >= 2.0
        return mapbox_vector_tile.decode(content, default_options={"y_coord_down": True})
    except TypeError:  # старые версии
        return mapbox_vector_tile.decode(content, y_coord_down=True)


def fetch_images(bbox, token, session=None, verbose=True):
    """Возвращает список снимков (dict со свойствами + lon/lat) внутри bbox."""
    s = session or requests.Session()
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y1 = lonlat_to_tile(min_lon, min_lat, Z)
    x1, y0 = lonlat_to_tile(max_lon, max_lat, Z)
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    out, t0 = [], time.time()
    for i, (x, y) in enumerate(tiles, 1):
        r = s.get(TILE_URL.format(z=Z, x=x, y=y), params={"access_token": token}, timeout=60)
        if r.status_code in (401, 403):
            sys.exit(f"\nОшибка авторизации {r.status_code}: проверьте токен.\n{r.text[:300]}")
        if r.status_code == 200 and r.content:
            layer = decode(r.content).get("image")
            if layer:
                extent = layer.get("extent", 4096)
                for f in layer["features"]:
                    px, py = f["geometry"]["coordinates"][:2]
                    lon, lat = tile_px_to_lonlat(x, y, Z, px, py, extent)
                    if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                        props = dict(f.get("properties", {}))
                        props.setdefault("id", f.get("id"))
                        props.update(lon=lon, lat=lat)
                        out.append(props)
        elif r.status_code not in (200, 204, 404):
            print(f"\n[!] Тайл {x}/{y}: HTTP {r.status_code} {r.text[:200]}")
        if verbose:
            print(f"\rТайл {i}/{len(tiles)} | найдено снимков: {len(out)} | {time.time() - t0:.0f} с",
                  end="", flush=True)
    if verbose:
        print()
    return out


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path.cwd() / ".env")
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token or len(sys.argv) != 5:
        sys.exit("Использование: python check_area.py MIN_LON MIN_LAT MAX_LON MAX_LAT (и токен в .env)")

    imgs = fetch_images(tuple(float(v) for v in sys.argv[1:]), token)
    years = Counter(datetime.fromtimestamp(i["captured_at"] / 1000, tz=timezone.utc).year
                    for i in imgs if i.get("captured_at"))
    print(f"\nВсего снимков: {len(imgs)}")
    print(f"Из них 360-панорам: {sum(bool(i.get('is_pano')) for i in imgs)}")
    print(f"Последовательностей (поездок/прогулок): {len({i.get('sequence_id') for i in imgs})}")
    print("По годам:", dict(sorted(years.items())))
    if imgs:
        print("Пример свойств снимка:", {k: imgs[0][k] for k in list(imgs[0])[:8]})
