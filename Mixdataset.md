# ReflectCities mixed_GSV Dataset Setup

This note describes how to organize `ReflectCities/mixed_GSV` on a new server when the server already has:

```text
/home/ittc402xu3/SSD_FAST_DATASETS/vpr_dataset/
  generated/
    images/
    metadata/
    summary.json
  GSVcities/
    Dataframes/
    Images/
```

The goal is to create a lightweight mixed dataset with symlinks instead of copying image files.

## 1. Target Structure

Use this root path on the new server:

```bash
ROOT=/home/ittc402xu3/SSD_FAST_DATASETS/vpr_dataset
```

Recommended final layout:

```text
vpr_dataset/
  generated/
    images/
    metadata/
    summary.json
  GSVcities/
    Dataframes/
    Images/
  ReflectCities/
    generated -> ../generated
    gsv/
      Dataframes/*.csv -> ../../GSVcities/Dataframes/*.csv
      Images/<city>    -> ../../GSVcities/Images/<city>
    mixed_GSV/
      Dataframes/*.csv
      Images/<city>/*.jpg -> real GSV image or generated image
```

Important:

- `generated/metadata/*.csv` is not the final mixed training CSV.
- `generated/metadata/*.csv` is generation metadata and must be converted into standard training rows.
- `mixed_GSV/Dataframes/*.csv` should contain both real GSV rows and synthetic generated rows.
- `mixed_GSV/Images/<city>/` should contain symlinks to both real GSV images and generated images.

Expected reference count for the current 23-city GSV setup:

```text
mixed_GSV/Dataframes CSV count: 23
mixed_GSV/Images symlink count: 570711
broken symlink count: 0
```

Create the basic directories and simple symlinks:

```bash
ROOT=/home/ittc402xu3/SSD_FAST_DATASETS/vpr_dataset
RC=$ROOT/ReflectCities

mkdir -p "$RC"
ln -sfn ../generated "$RC/generated"

mkdir -p "$RC/gsv/Dataframes" "$RC/gsv/Images"

for f in "$ROOT/GSVcities/Dataframes/"*.csv; do
  ln -sfn "$f" "$RC/gsv/Dataframes/$(basename "$f")"
done

for d in "$ROOT/GSVcities/Images/"*; do
  [ -d "$d" ] && ln -sfn "$d" "$RC/gsv/Images/$(basename "$d")"
done
```

## 2. Build mixed_GSV

`mixed_GSV/Dataframes/*.csv` is:

```text
GSVcities/Dataframes/*.csv
+ synthetic rows converted from generated/metadata/*.csv
```

`mixed_GSV/Images/<city>/` is:

```text
real GSV image symlinks
+ generated image symlinks
```

Save the following script as `build_mixed_gsv.py` on the new server:

```python
#!/usr/bin/env python3
import csv
import os
from pathlib import Path

ROOT = Path("/home/ittc402xu3/SSD_FAST_DATASETS/vpr_dataset")

GSV = ROOT / "GSVcities"
GEN = ROOT / "generated"
OUT = ROOT / "ReflectCities" / "mixed_GSV"

OUT_DF = OUT / "Dataframes"
OUT_IMG = OUT / "Images"

OUT_DF.mkdir(parents=True, exist_ok=True)
OUT_IMG.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "place_id", "year", "month", "northdeg",
    "city_id", "lat", "lon", "panoid", "similarity",
]


def link_force(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def panoid_from_dedup_filename(row):
    # dedup_filename example:
    # Bangkok_0000001_2017_05_412_13.715..._100.484..._<source_panoid>__reflectvpr_dual_rain_vehicle.jpg
    name = Path(row["dedup_filename"]).name
    stem = name[:-4] if name.endswith(".jpg") else name
    if "__reflectvpr_" not in stem:
        return row["source_panoid"]

    base, suffix = stem.split("__reflectvpr_", 1)
    prefix = (
        f'{row["city"]}_'
        f'{int(row["source_place_id"]):07d}_'
        f'{int(row["source_year"]):04d}_'
        f'{int(row["source_month"]):02d}_'
        f'{int(row["source_northdeg"]):03d}_'
        f'{row["source_lat"]}_'
        f'{row["source_lon"]}_'
    )
    if not base.startswith(prefix):
        return row["source_panoid"]

    return base[len(prefix):] + "__reflectvpr_" + suffix


for meta_csv in sorted((GEN / "metadata").glob("*.csv")):
    city = meta_csv.stem
    real_csv = GSV / "Dataframes" / f"{city}.csv"

    if not real_csv.exists():
        print(f"[skip] no GSV csv for {city}: {real_csv}")
        continue

    rows = []

    # 1. Real GSV rows and real image symlinks.
    with real_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: r[k] for k in COLUMNS})

            real_name = (
                f'{r["city_id"]}_{int(r["place_id"]):07d}_'
                f'{int(r["year"]):04d}_{int(r["month"]):02d}_{int(r["northdeg"]):03d}_'
                f'{r["lat"]}_{r["lon"]}_{r["panoid"]}.jpg'
            )
            src = GSV / "Images" / city / real_name
            dst = OUT_IMG / city / real_name
            if src.exists():
                link_force(src, dst)
            else:
                print(f"[missing real] {src}")

    # 2. Synthetic rows converted from generated metadata.
    with meta_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("passed", "True") != "True":
                continue

            gen_src = GEN / "images" / r["city"] / r["route"] / r["dedup_filename"]
            if not gen_src.exists():
                print(f"[missing generated] {gen_src}")
                continue

            synthetic_panoid = panoid_from_dedup_filename(r)

            rows.append({
                "place_id": r["source_place_id"],
                "year": str(int(r["source_year"])),
                "month": str(int(r["source_month"])),
                "northdeg": str(int(r["source_northdeg"])),
                "city_id": r["city"],
                "lat": r["source_lat"],
                "lon": r["source_lon"],
                "panoid": synthetic_panoid,
                "similarity": r.get("s_geo", "1.0"),
            })

            link_force(gen_src, OUT_IMG / city / r["dedup_filename"])

    out_csv = OUT_DF / f"{city}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] {city}: {len(rows)} rows -> {out_csv}")
```

Run it:

```bash
python3 build_mixed_gsv.py
```

Use the resulting dataset in training:

```bash
--training_dataset /home/ittc402xu3/SSD_FAST_DATASETS/vpr_dataset/ReflectCities/mixed_GSV
```

## 3. Final Checks

Check CSV count:

```bash
ROOT=/home/ittc402xu3/SSD_FAST_DATASETS/vpr_dataset
RC=$ROOT/ReflectCities/mixed_GSV

echo "=== CSV count ==="
find "$RC/Dataframes" -name '*.csv' | wc -l
```

Check image symlink count:

```bash
echo "=== image symlink count ==="
find "$RC/Images" -type l | wc -l
```

Fast broken-link check. This prints only the first broken link, if any:

```bash
echo "=== first broken symlink, if any ==="
find "$RC/Images" -type l ! -exec test -e {} \; -print -quit
```

Full broken-link check. This can take longer because it scans all image symlinks:

```bash
echo "=== broken symlinks, first 10 ==="
find "$RC/Images" -type l ! -exec test -e {} \; -print 2>/dev/null | head
```

Expected output for the current GSV mixed dataset:

```text
=== CSV count ===
23
=== image symlink count ===
570711
=== first broken symlink, if any ===

```

No output after `first broken symlink` means no broken symlink was found.
