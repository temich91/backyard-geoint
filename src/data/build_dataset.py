#!/usr/bin/env python3
"""
build_dataset.py — единый датасет для геолокации по фото из снимков Mapillary.

Снимки скачиваются ОДИН раз, а дальше из них строятся «представления» (CSV-файлы) под любой
вариант проекта — пересоздавать датасет не придётся:

  views/database.csv  — база для поиска (FAISS): плотные снимки во всех направлениях.
  views/queries.csv   — запросы для проверки: снимки из ДРУГИХ поездок, чем база.
  views/train.csv,    — обучающий набор, если сеть придётся учить с нуля:
  views/val.csv,        класс = ячейка ~50x50 м + направление (С/В/Ю/З).
  views/classes.csv     Это подмножество тех же файлов, что и база.

Работает и для целого города: снимки обрабатываются потоком, в памяти остаются только
кандидаты по ячейкам. Тайлы кэшируются в out/tiles, повторные запуски их не перекачивают.

Главное правило: все поездки (sequence) заранее делятся на две непересекающиеся группы.
  - «опорные» (ref) — из них берутся база и обучающий набор;
  - «запросные» (query) — только для проверки, в обучение и базу не попадают никогда.
Поэтому любая оценка (готовая модель, своя сеть, SIFT + RANSAC) будет честной.

Использование:
  # 1. Посмотреть объём, ничего не скачивая
  python build_dataset.py --bbox 56.046 54.758 56.087 54.783 --dry-run
  # 2. Скачать (можно прерывать: повторный запуск докачает)
  python build_dataset.py --bbox 56.046 54.758 56.087 54.783
  # 3. Позже поменять разбиение на классы БЕЗ скачивания
  python build_dataset.py --bbox 56.046 54.758 56.087 54.783 --views-only --class-cell 75

Пространственный вариант (обучение и проверка на разных территориях):
  --exclude-bbox  не брать снимки из области (тестового района) + --exclude-buffer метров вокруг;
  --spatial-views дополнительно собрать views_spatial/: train и val из разных блоков
                  --val-block метров, с полосой --val-buffer между ними (см. README.md).

Установка:  pip install requests numpy opencv-python mapbox-vector-tile tqdm
Токен:      файл .env рядом со скриптом: MAPILLARY_TOKEN=MLY|...
Рядом должен лежать check_area.py.
Лицензия снимков Mapillary: CC BY-SA 4.0 (автор — creator_id в metadata.csv).
"""

import argparse
import csv
import hashlib
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_area import TILE_URL, Z, decode, load_dotenv, lonlat_to_tile, tile_px_to_lonlat  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

ENTITY_URL = "https://graph.mapillary.com/{id}"
SECTORS = ["N", "E", "S", "W"]
AVG_KB = 60          # примерный размер одного сохранённого фото при --max-side 512
REF_CANDIDATES = 6   # сколько поездок-кандидатов помнить на ячейку базы (из них выбираются разнесённые по времени)
QUERY_CANDIDATES = 4 # сколько кандидатов помнить на ячейку запросов


# ============================================================================ геометрия

class Grid:
    """Перевод координат в метры относительно юго-западного угла bbox."""

    def __init__(self, bbox):
        self.min_lon, self.min_lat = bbox[0], bbox[1]
        lat0 = math.radians((bbox[1] + bbox[3]) / 2)
        self.m_lat, self.m_lon = 110_540.0, 111_320.0 * math.cos(lat0)

    def xy(self, lat, lon):
        return (lon - self.min_lon) * self.m_lon, (lat - self.min_lat) * self.m_lat

    def cell(self, lat, lon, size):
        x, y = self.xy(lat, lon)
        return int(x // size), int(y // size)

    def center(self, ix, iy, size):
        return (self.min_lat + (iy + 0.5) * size / self.m_lat,
                self.min_lon + (ix + 0.5) * size / self.m_lon)


def sector(h):
    return SECTORS[int(((h + 45) % 360) // 90)]


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def spread_pick(items, k):
    if len(items) <= k:
        return list(items)
    idx = sorted(set(np.linspace(0, len(items) - 1, k).round().astype(int)))
    return [items[i] for i in idx]


# ============================================================================ чтение тайлов

class Img(NamedTuple):
    """Компактная запись о снимке (кортеж занимает в разы меньше памяти, чем dict)."""
    id: int
    lat: float
    lon: float
    heading: float
    captured_at: int
    seq: str
    creator: int
    quality: float


def iter_tiles(bbox, token, cache_dir, workers):
    """Скачивает векторные тайлы (с кэшем на диске) и отдаёт их по мере готовности."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y1 = lonlat_to_tile(min_lon, min_lat, Z)
    x1, y0 = lonlat_to_tile(max_lon, max_lat, Z)
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    sess = make_session(workers)

    def get(t):
        x, y = t
        p = cache_dir / f"{x}_{y}.pbf"
        if p.exists():
            return t, p.read_bytes()
        r = sess.get(TILE_URL.format(z=Z, x=x, y=y), params={"access_token": token}, timeout=120)
        if r.status_code in (401, 403):
            raise PermissionError(f"Ошибка авторизации {r.status_code}: проверьте токен. {r.text[:200]}")
        if r.status_code == 200:
            data = r.content
        elif r.status_code in (204, 404):
            data = b""
        else:
            raise RuntimeError(f"тайл {x}/{y}: HTTP {r.status_code}")
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        return t, data

    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(get, t) for t in tiles]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Тайлы", unit="тайл"):
            try:
                yield fut.result()
            except PermissionError as e:
                sys.exit(str(e))
            except Exception as e:
                failed += 1
                print(f"\n[!] {e}")
    if failed:
        print(f"[!] Не загрузилось тайлов: {failed}. Запустите ещё раз — они докачаются.")


def tile_points(x, y, data, bbox):
    if not data:
        return
    layer = decode(data).get("image")
    if not layer:
        return
    extent = layer.get("extent", 4096)
    min_lon, min_lat, max_lon, max_lat = bbox
    for f in layer["features"]:
        px, py = f["geometry"]["coordinates"][:2]
        lon, lat = tile_px_to_lonlat(x, y, Z, px, py, extent)
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            props = f.get("properties", {})
            if "id" not in props and f.get("id") is not None:
                props["id"] = f["id"]
            yield props, lat, lon


# ============================================================================ отбор

def role_of(seq, seed, query_frac):
    """Роль поездки определяется хэшем её id — одинаково при любом запуске и любом bbox."""
    h = int.from_bytes(hashlib.md5(f"{seed}:{seq}".encode()).digest()[:8], "big")
    return "query" if h / 2 ** 64 < query_frac else "ref"


def scan(args, grid, out):
    """Один проход по всем снимкам: фильтрация, роли и кандидаты по ячейкам."""
    stats = Counter()
    seqs = {"ref": set(), "query": set()}
    ref_cells, q_cells = {}, {}
    seen = Counter()
    rng = random.Random(args.seed)

    ex = args.exclude_box
    for (x, y), data in iter_tiles(tuple(args.bbox), args.token, out / "tiles", args.workers):
        for p, lat, lon in tile_points(x, y, data, args.bbox):
            stats["всего"] += 1
            if ex and ex[0] <= lon <= ex[2] and ex[1] <= lat <= ex[3]:
                stats["исключённая область"] += 1
                continue
            if p.get("is_pano"):
                stats["панорамы"] += 1
                continue
            q = p.get("quality_score")
            if q is not None and q < args.min_quality:
                stats["низкое качество"] += 1
                continue
            h = p.get("compass_angle")
            if h is None or (h == 0.0 and not args.keep_zero_heading):
                stats["нет направления"] += 1
                continue
            seq = p.get("sequence_id")
            if not seq or p.get("id") is None:
                stats["нет поездки"] += 1
                continue
            stats["прошли фильтры"] += 1

            role = role_of(seq, args.seed, args.query_frac)
            seqs[role].add(seq)
            img = Img(int(p["id"]), lat, lon, float(h), int(p.get("captured_at") or 0), seq,
                      int(p.get("creator_id") or 0), float(q if q is not None else -1))

            if role == "ref":
                key = ("r", *grid.cell(lat, lon, args.db_cell), sector(h))
                d = ref_cells.setdefault(key, {})
                if seq in d:
                    continue
                seen[key] += 1
                if len(d) < REF_CANDIDATES:
                    d[seq] = img
                elif rng.randrange(seen[key]) < REF_CANDIDATES:   # резервуарная выборка поездок
                    del d[rng.choice(list(d))]
                    d[seq] = img
            else:
                key = ("q", *grid.cell(lat, lon, args.query_cell), sector(h))
                lst = q_cells.setdefault(key, [])
                seen[key] += 1
                if len(lst) < QUERY_CANDIDATES:
                    lst.append(img)
                else:
                    j = rng.randrange(seen[key])
                    if j < QUERY_CANDIDATES:
                        lst[j] = img
    return ref_cells, q_cells, stats, seqs


def select_ref(ref_cells, args):
    """В каждой ячейке и направлении — до per_cell снимков из разных поездок, разнесённых по времени."""
    chosen = []
    for d in ref_cells.values():
        imgs = sorted(d.values(), key=lambda i: i.captured_at)
        chosen.extend(spread_pick(imgs, args.per_cell))
    return chosen


def select_queries(q_cells, ref_sel, grid, args):
    """До одного запроса на ячейку и направление; только там, где в базе есть правильный ответ."""
    r = args.pos_radius
    index = defaultdict(list)
    for img in ref_sel:
        index[grid.cell(img.lat, img.lon, r)].append(img)

    def n_positives(q):
        cx, cy = grid.cell(q.lat, q.lon, r)
        qx, qy = grid.xy(q.lat, q.lon)
        n = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for d in index.get((cx + dx, cy + dy), []):
                    x, y = grid.xy(d.lat, d.lon)
                    if math.hypot(x - qx, y - qy) <= r and angle_diff(d.heading, q.heading) <= args.pos_angle:
                        n += 1
        return n

    rng = random.Random(args.seed + 2)
    picked = []
    for key in sorted(q_cells):
        cands = list(q_cells[key])
        rng.shuffle(cands)
        for q in cands:
            n = n_positives(q)
            if n > 0:
                picked.append((q, n))
                break
    rng.shuffle(picked)
    return picked[: args.n_queries]


def to_row(img, n_pos=""):
    return {"id": img.id, "lat": img.lat, "lon": img.lon, "compass_angle": img.heading,
            "captured_at": img.captured_at or None, "sequence_id": img.seq,
            "creator_id": img.creator, "quality_score": img.quality if img.quality >= 0 else "",
            "n_positives": n_pos}


# ============================================================================ представления

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_views(out, grid, args):
    meta_path = out / "metadata.csv"
    if not meta_path.exists():
        sys.exit("Нет metadata.csv — сначала скачайте датасет.")
    with open(meta_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (out / r["path"]).exists()]
    for r in rows:
        r["lat"], r["lon"], r["compass_angle"] = float(r["lat"]), float(r["lon"]), float(r["compass_angle"])

    ref = [r for r in rows if r["role"] == "ref"]
    qry = [r for r in rows if r["role"] == "query"]
    base_fields = ["path", "image_id", "lat", "lon", "compass_angle", "captured_at", "sequence_id"]

    # --- база
    write_csv(out / "views" / "database.csv", ref, base_fields)

    # --- обучающий набор: класс = class_cell x направление
    def class_of(r):
        ix, iy = grid.cell(r["lat"], r["lon"], args.class_cell)
        return f"c{iy:03d}_{ix:03d}_{sector(r['compass_angle'])}", ix, iy

    groups = defaultdict(list)
    for r in ref:
        name, ix, iy = class_of(r)
        r["class"] = name
        groups[name].append(r)

    rng = random.Random(args.seed + 3)
    classes, train, val = [], [], []
    for name in sorted(groups):
        members = groups[name]
        seqs = sorted({m["sequence_id"] for m in members})
        if len(seqs) < args.min_sequences:
            continue
        rng.shuffle(seqs)
        n_val = max(1, round(len(seqs) * args.val_frac))
        val_seqs = set(seqs[:n_val])
        tr = [m for m in members if m["sequence_id"] not in val_seqs]
        va = [m for m in members if m["sequence_id"] in val_seqs]
        train += tr
        val += va
        _, ix, iy = class_of(members[0])
        lat_c, lon_c = grid.center(ix, iy, args.class_cell)
        classes.append({"class": name, "class_idx": len(classes), "center_lat": f"{lat_c:.6f}",
                        "center_lon": f"{lon_c:.6f}", "direction": name[-1],
                        "n_train": len(tr), "n_val": len(va), "n_sequences": len(seqs)})

    idx = {c["class"]: c["class_idx"] for c in classes}
    for r in train + val:
        r["class_idx"] = idx[r["class"]]
    cls_fields = base_fields + ["class", "class_idx"]
    write_csv(out / "views" / "train.csv", train, cls_fields)
    write_csv(out / "views" / "val.csv", val, cls_fields)
    write_csv(out / "views" / "classes.csv", classes, list(classes[0]) if classes else ["class"])

    # --- запросы (+ их класс, если такой класс есть в обучении — для оценки классификатора)
    for q in qry:
        name = class_of(q)[0]
        q["class"] = name if name in idx else ""
        q["class_idx"] = idx.get(name, "")
    write_csv(out / "views" / "queries.csv", qry, base_fields + ["n_positives", "class", "class_idx"])

    print(f"\nПредставления в {out / 'views'}:")
    print(f"  database.csv : {len(ref)} снимков")
    print(f"  queries.csv  : {len(qry)} запросов "
          f"(из них с классом из обучения: {sum(1 for q in qry if q['class'])})")
    print(f"  train/val    : {len(classes)} классов, {len(train)} / {len(val)} снимков "
          f"(ячейка {args.class_cell:g} м, >= {args.min_sequences} поездок)")

    if args.spatial_views:
        build_spatial_views(out, grid, args, ref, qry, class_of, base_fields)


def block_splitter(grid, args):
    """Пространственный сплит: квадраты val_block x val_block м целиком уходят в train или val.

    Возвращает функцию (lat, lon) -> 'train' | 'val' | None. None — снимок ближе val_buffer
    к границе между train и val: такие выбрасываются, чтобы по разные стороны границы
    не оказались снимки одних и тех же домов.
    """
    size, buf = args.val_block, args.val_buffer
    cache = {}

    def block(bx, by):
        if (bx, by) not in cache:
            h = int.from_bytes(hashlib.md5(f"{args.seed}:block:{bx}:{by}".encode()).digest()[:8], "big")
            cache[bx, by] = "val" if h / 2 ** 64 < args.val_frac else "train"
        return cache[bx, by]

    def split_of(lat, lon):
        x, y = grid.xy(lat, lon)
        found = {block(bx, by)
                 for bx in range(math.floor((x - buf) / size), math.floor((x + buf) / size) + 1)
                 for by in range(math.floor((y - buf) / size), math.floor((y + buf) / size) + 1)}
        return found.pop() if len(found) == 1 else None

    return split_of


def build_spatial_views(out, grid, args, ref, qry, class_of, base_fields):
    """views_spatial/: train и val из разных частей территории (а не только из разных поездок)."""
    if args.val_buffer < args.pos_radius:
        print(f"[!] --val-buffer {args.val_buffer:g} м меньше --pos-radius {args.pos_radius:g} м: "
              f"правильные ответы для запросов val могут оказаться в train-блоках.")
    split_of = block_splitter(grid, args)
    dropped = Counter()

    groups = defaultdict(list)
    val_db = []
    for r in ref:
        s = split_of(r["lat"], r["lon"])
        if s is None:
            dropped["база"] += 1
            continue
        r = dict(r)
        if s == "val":
            val_db.append(r)
        groups[s, class_of(r)[0]].append(r)

    classes, rows = [], {"train": [], "val": []}
    for s in ("train", "val"):
        for _, name in sorted(k for k in groups if k[0] == s):
            members = groups[s, name]
            n_seq = len({m["sequence_id"] for m in members})
            if n_seq < args.min_sequences:
                continue
            _, ix, iy = class_of(members[0])
            lat_c, lon_c = grid.center(ix, iy, args.class_cell)
            idx = len(classes)
            classes.append({"class": name, "class_idx": idx, "split": s, "center_lat": f"{lat_c:.6f}",
                            "center_lon": f"{lon_c:.6f}", "direction": name[-1],
                            "n_images": len(members), "n_sequences": n_seq})
            for m in members:
                m["class"], m["class_idx"] = name, idx
                rows[s].append(m)

    # Правильные ответы считаются заново только по val_database: часть ответов могла выпасть в буфер
    rad = args.pos_radius
    index = defaultdict(list)
    for d in val_db:
        index[grid.cell(d["lat"], d["lon"], rad)].append((*grid.xy(d["lat"], d["lon"]), d["compass_angle"]))

    def n_positives(q):
        qx, qy = grid.xy(q["lat"], q["lon"])
        cx, cy = grid.cell(q["lat"], q["lon"], rad)
        return sum(1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for x, y, h in index.get((cx + dx, cy + dy), [])
                   if math.hypot(x - qx, y - qy) <= rad and angle_diff(h, q["compass_angle"]) <= args.pos_angle)

    val_q = []
    for q in qry:
        s = split_of(q["lat"], q["lon"])
        if s is None:
            dropped["запросы"] += 1
        elif s == "val":
            n = n_positives(q)
            if n == 0:
                dropped["запросы без ответа"] += 1
                continue
            val_q.append({**q, "n_positives": n})

    vdir = out / "views_spatial"
    cls_fields = base_fields + ["class", "class_idx"]
    write_csv(vdir / "train.csv", rows["train"], cls_fields)
    write_csv(vdir / "val.csv", rows["val"], cls_fields)
    write_csv(vdir / "classes.csv", classes, list(classes[0]) if classes else ["class"])
    write_csv(vdir / "val_database.csv", val_db, base_fields)
    write_csv(vdir / "val_queries.csv", val_q, base_fields + ["n_positives"])

    n_cls = Counter(c["split"] for c in classes)
    print(f"\nПространственный сплит в {vdir} (блоки {args.val_block:g} м, доля val {args.val_frac:g}, "
          f"буфер {args.val_buffer:g} м):")
    print(f"  train.csv    : {len(rows['train'])} снимков, {n_cls['train']} классов")
    print(f"  val.csv      : {len(rows['val'])} снимков, {n_cls['val']} классов")
    print(f"  val_database : {len(val_db)} снимков, val_queries: {len(val_q)} запросов")
    print(f"  выброшено в буфере: база {dropped['база']}, запросы {dropped['запросы']}; "
          f"запросов val без ответа в val_database: {dropped['запросы без ответа']}")


# ============================================================================ загрузка

def make_session(workers):
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=max(10, workers * 2)))
    return s


def image_path(out, image_id):
    image_id = str(image_id)
    return out / "images" / image_id[-2:] / f"{image_id}.jpg"


def get_thumb_url(api, image_id, token):
    """Ссылка на фото. Токен передаётся в параметрах запроса (так API принимает его надёжно)."""
    r = api.get(ENTITY_URL.format(id=image_id),
                params={"fields": "thumb_1024_url", "access_token": token}, timeout=60)
    if r.status_code != 200:
        # Текст ответа без URL, чтобы токен не попал в логи
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    url = r.json().get("thumb_1024_url")
    if not url:
        raise RuntimeError(f"нет ссылки на изображение в ответе: {r.text[:200]}")
    return url


def download_one(img, path, api, cdn, args):
    if path.exists():
        return
    url = get_thumb_url(api, img["id"], args.token)
    r = cdn.get(url, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"CDN HTTP {r.status_code}")
    im = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError("не удалось декодировать")
    h, w = im.shape[:2]
    scale = args.max_side / max(h, w)
    if scale < 1:
        im = cv2.resize(im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    os.replace(tmp, path)  # чтобы прерванная запись не оставила битый файл


def download_all(plan, out, args):
    api = make_session(args.workers)
    cdn = make_session(args.workers)

    # Пробный запрос: если API отвечает ошибкой, останавливаемся сразу, а не через 20 минут
    todo = [img for _, img in plan if not image_path(out, img["id"]).exists()]
    if todo:
        try:
            get_thumb_url(api, todo[0]["id"], args.token)
        except Exception as e:
            sys.exit(f"Пробный запрос к API не прошёл, загрузка не начата.\nОтвет сервера: {e}")

    rows, failed = [], []
    fail_streak = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, img, image_path(out, img["id"]), api, cdn, args): (role, img)
                   for role, img in plan}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Загрузка", unit="фото"):
            role, img = futures[fut]
            try:
                fut.result()
                fail_streak = 0
            except Exception as e:
                failed.append(f"{img['id']}\t{e}")
                fail_streak += 1
                if fail_streak >= 200:
                    for f in futures:
                        f.cancel()
                    print(f"\n[!] 200 ошибок подряд, загрузка остановлена. Последняя: {e}")
                    break
                continue
            ts = img.get("captured_at")
            rows.append({
                "path": image_path(out, img["id"]).relative_to(out).as_posix(),
                "image_id": img["id"], "role": role,
                "lat": f"{img['lat']:.7f}", "lon": f"{img['lon']:.7f}",
                "compass_angle": f"{img['compass_angle']:.2f}",
                "captured_at": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if ts else "",
                "sequence_id": img["sequence_id"], "creator_id": img.get("creator_id", ""),
                "quality_score": img.get("quality_score", ""), "n_positives": img.get("n_positives", ""),
            })
    rows.sort(key=lambda r: (r["role"], str(r["image_id"])))
    write_csv(out / "metadata.csv", rows, list(rows[0]) if rows else ["path"])
    if failed:
        (out / "failed.txt").write_text("\n".join(failed), encoding="utf-8")
    else:
        (out / "failed.txt").unlink(missing_ok=True)  # не оставлять список ошибок от прошлого запуска
    print(f"\nСкачано: {len(rows)} фото" + (f", ошибок: {len(failed)} (повторный запуск докачает)"
                                           if failed else ""))


# ============================================================================ main

def parse_args():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path.cwd() / ".env")
    ap = argparse.ArgumentParser(description="Единый датасет для геолокации по фото (Mapillary)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--out", type=Path, default=Path("data") / "geo")
    ap.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN"))

    g = ap.add_argument_group("отбор снимков (влияет на то, что скачивается)")
    g.add_argument("--min-quality", type=float, default=0.5)
    g.add_argument("--keep-zero-heading", action="store_true",
                   help="Не выкидывать снимки с направлением ровно 0° (часто это 'неизвестно')")
    g.add_argument("--query-frac", type=float, default=0.2, help="Доля поездок под запросы")
    g.add_argument("--db-cell", type=float, default=15.0, help="Шаг базы, м")
    g.add_argument("--per-cell", type=int, default=2,
                   help="Снимков базы на ячейку и направление (из разных поездок)")
    g.add_argument("--n-queries", type=int, default=1000)
    g.add_argument("--query-cell", type=float, default=25.0, help="Не больше 1 запроса на такую ячейку")
    g.add_argument("--pos-radius", type=float, default=25.0,
                   help="Запрос считается найденным, если ответ ближе этого расстояния, м")
    g.add_argument("--pos-angle", type=float, default=90.0,
                   help="... и отличается по направлению не больше чем на столько градусов")
    g.add_argument("--max-side", type=int, default=512)
    g.add_argument("--exclude-bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                   help="Не брать снимки из этой области (например, тестового района)")
    g.add_argument("--exclude-buffer", type=float, default=100.0,
                   help="Расширить --exclude-bbox на столько метров во все стороны")

    v = ap.add_argument_group("обучающий набор (можно менять потом через --views-only)")
    v.add_argument("--class-cell", type=float, default=50.0, help="Размер ячейки класса, м")
    v.add_argument("--min-sequences", type=int, default=4, help="Минимум разных поездок в классе")
    v.add_argument("--val-frac", type=float, default=0.15,
                   help="Доля поездок под val (views) или доля блоков под val (views_spatial)")
    v.add_argument("--spatial-views", action="store_true",
                   help="Дополнительно собрать views_spatial/: train и val из разных частей территории")
    v.add_argument("--val-block", type=float, default=500.0, help="Размер блока пространственного сплита, м")
    v.add_argument("--val-buffer", type=float, default=50.0,
                   help="Снимки ближе этого к границе train/val выбрасываются, м")

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Только оценить объём")
    ap.add_argument("--views-only", action="store_true", help="Пересобрать CSV без скачивания")
    ap.add_argument("--refresh", action="store_true", help="Заново скачать тайлы (сбросить кэш)")
    args = ap.parse_args()
    if not (args.bbox[0] < args.bbox[2] and args.bbox[1] < args.bbox[3]):
        ap.error("bbox: ожидается MIN_LON MIN_LAT MAX_LON MAX_LAT")
    if not args.views_only and not args.token:
        ap.error("нужен токен: MAPILLARY_TOKEN в .env или --token")
    args.exclude_box = None
    if args.exclude_bbox:
        x0, y0, x1, y1 = args.exclude_bbox
        if not (x0 < x1 and y0 < y1):
            ap.error("exclude-bbox: ожидается MIN_LON MIN_LAT MAX_LON MAX_LAT")
        dlat = args.exclude_buffer / 110_540.0
        dlon = args.exclude_buffer / (111_320.0 * math.cos(math.radians((y0 + y1) / 2)))
        args.exclude_box = (x0 - dlon, y0 - dlat, x1 + dlon, y1 + dlat)
    return args


def main():
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid(args.bbox)

    if args.views_only:
        build_views(out, grid, args)
        return

    # 1. Один проход по тайлам (тайлы кэшируются в out/tiles)
    if args.refresh and (out / "tiles").exists():
        shutil.rmtree(out / "tiles")
    ref_cells, q_cells, stats, seqs = scan(args, grid, out)
    print("Фильтрация:", ", ".join(f"{k}: {v}" for k, v in stats.items()))
    print(f"Поездок: {len(seqs['ref']) + len(seqs['query'])} "
          f"(опорных {len(seqs['ref'])}, запросных {len(seqs['query'])})")

    # 2. Отбор
    ref_sel = select_ref(ref_cells, args)
    del ref_cells
    q_sel = select_queries(q_cells, ref_sel, grid, args)
    total = len(ref_sel) + len(q_sel)
    print(f"База: {len(ref_sel)} снимков (шаг {args.db_cell:g} м, до {args.per_cell} на ячейку и направление)")
    print(f"Запросы: {len(q_sel)}")
    print(f"Итого к загрузке: {total} фото, примерно {total * AVG_KB / 1024 ** 2:.1f} ГБ "
          f"(~{total / 10 / 3600:.1f} ч при скорости ~10 фото/с)")

    if args.dry_run:
        print("\n--dry-run: ничего не скачано.")
        return

    # 3. Загрузка и представления
    download_all([("ref", to_row(i)) for i in ref_sel] + [("query", to_row(q, n)) for q, n in q_sel],
                 out, args)
    build_views(out, grid, args)


if __name__ == "__main__":
    main()
