"""Download a single BlenderKit asset by asset_base_id + asset_type.

Usage:
  python data/blenderkit_download.py asset_base_id:<uuid> asset_type:<scene|model|material|brush|hdr>

Example:
  python data/blenderkit_download.py \\
    asset_base_id:e894abd6-28d8-4e99-bfb8-0ffa84e196a0 asset_type:scene

Requires BLENDERKIT_API_KEY in env (get from https://www.blenderkit.com/profile/).
Scenes/models download to data/scenes/ and data/objects/ respectively.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

API_BASE = "https://www.blenderkit.com/api/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent
BLENDER_BIN_DEFAULT = "/home/nas5/jungwooahn/projects/DronePhotographer/blender/blender"
INSPECT_SCRIPT = Path(__file__).parent / "blender_inspect.py"

ASSET_TYPE_TO_DIR = {
    "scene":    REPO_ROOT / "data/scenes",
    "model":    REPO_ROOT / "data/objects",
    "material": REPO_ROOT / "data/materials",
    "hdr":      REPO_ROOT / "data/lighting",
    "brush":    REPO_ROOT / "data/brushes",
}


def parse_kv_args(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in argv:
        if ":" not in a:
            raise SystemExit(f"Bad arg (expected key:value): {a!r}")
        k, v = a.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def search_asset(asset_base_id: str, asset_type: str, token: str) -> dict:
    url = f"{API_BASE}/search/"
    params = {"query": f"asset_base_id:{asset_base_id} asset_type:{asset_type}"}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise SystemExit(f"No results for {asset_base_id} ({asset_type})")
    return results[0]


def pick_blend_file(asset: dict) -> dict:
    """Pick the highest-resolution blend file from an asset's files list."""
    files = asset.get("files") or []
    blends = [f for f in files if f.get("fileType", "").startswith("blend")]
    if not blends:
        raise SystemExit(f"No blend file in asset {asset.get('name')!r}")
    # Prefer 'blend' over 'blend_*' resolution variants; fall back to first
    exact = [f for f in blends if f["fileType"] == "blend"]
    return (exact or blends)[0]


def resolve_download_url(resolver_url: str, token: str) -> str:
    """BlenderKit's download URLs are API endpoints that return the real S3 URL.

    Calls the resolver with auth + required params (scene_uuid, addon_version)
    and returns the file_path to actually download from.
    """
    import uuid as _uuid
    params = {
        "scene_uuid": str(_uuid.uuid4()),
        "addon_version": "3.12.0",
    }
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(resolver_url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    real_url = data.get("file_path") or data.get("filePath")
    if not real_url:
        raise SystemExit(f"Resolver returned no file_path: {data}")
    return real_url


def download(url: str, dest: Path, token: str) -> None:
    # If the URL is a BlenderKit API resolver, exchange it for the real S3 URL first.
    if "blenderkit.com/api/" in url:
        url = resolve_download_url(url, token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # Don't send bearer to S3/CDN — strip auth on non-blenderkit hosts
    if "blenderkit.com" not in url:
        headers = {}
    with requests.get(url, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            total = 0
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                total += len(chunk)
                print(f"\r  downloaded {total / 1e6:.1f} MB", end="", flush=True)
            print()


def _is_blend(path: Path) -> bool:
    """BlenderKit .blend files are either raw BLENDER magic or zstd-compressed."""
    with open(path, "rb") as f:
        head = f.read(12)
    if head.startswith(b"BLENDER"):
        return True
    # Zstandard magic bytes: 0x28 0xB5 0x2F 0xFD
    return head[:4] == b"\x28\xB5\x2F\xFD"


def extract_if_zip(path: Path, into: Path, target_stem: str) -> list[Path]:
    """If path is a zip, extract into `into/`. If it's a blend (raw or zstd-
    compressed), rename to `<target_stem>.blend`. Otherwise leave as-is.

    After a zip extraction, the primary .blend is also renamed to match
    `target_stem` so its basename matches the enclosing directory.
    """
    if zipfile.is_zipfile(path):
        into.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(into)
        path.unlink()
        blends = sorted(into.rglob("*.blend"))
        # Rename the largest .blend at the top level to match target_stem
        top_level = [b for b in blends if b.parent == into]
        if top_level:
            primary = max(top_level, key=lambda p: p.stat().st_size)
            desired = into / f"{target_stem}.blend"
            if primary != desired:
                primary.rename(desired)
                blends = sorted(into.rglob("*.blend"))
        return blends
    if _is_blend(path):
        desired = path.parent / f"{target_stem}.blend"
        if path != desired:
            path.rename(desired)
            return [desired]
        return [path]
    return [path]


def spawn_background_inspect(blend_path: Path, blender_bin: str = BLENDER_BIN_DEFAULT) -> None:
    """Launch blender_inspect.py on a .blend in the background.

    Writes the ###INSPECT### JSON line to <blend_dir>/inspection.json (same
    dir as the .blend). Does not block: the caller returns immediately.
    """
    report_path = blend_path.parent / "inspection.json"
    log_path = blend_path.parent / "inspection.stdout.log"
    # Run blender, pipe stdout to a script that extracts the verdict
    cmd = (
        f"{blender_bin} --background {blend_path!s} --python {INSPECT_SCRIPT!s} "
        f"2>/dev/null | grep '^###INSPECT###' | sed 's/^###INSPECT### //' "
        f"> {report_path!s} 2> {log_path!s}"
    )
    subprocess.Popen(
        ["bash", "-c", cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach from parent
    )
    print(f"  (inspection running in background → {report_path})")


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("-h", "--help", action="store_true")
    known, kv_argv = ap.parse_known_args()
    if known.help or not kv_argv:
        print(__doc__)
        sys.exit(0 if known.help else 1)

    kv = parse_kv_args(kv_argv)
    asset_base_id = kv.get("asset_base_id") or sys.exit("asset_base_id:<uuid> required")
    asset_type = kv.get("asset_type") or sys.exit("asset_type:<type> required")
    if asset_type not in ASSET_TYPE_TO_DIR:
        sys.exit(f"Unknown asset_type: {asset_type}. Pick one of: {list(ASSET_TYPE_TO_DIR)}")

    token = os.environ.get("BLENDERKIT_API_KEY", "")
    if not token:
        print("WARN: BLENDERKIT_API_KEY not set — only anonymous assets will work",
              file=sys.stderr)

    print(f"Searching BlenderKit for {asset_base_id} ({asset_type})...")
    asset = search_asset(asset_base_id, asset_type, token)
    asset_name = asset.get("name", asset_base_id)
    print(f"Found: {asset_name!r}")

    blend_file = pick_blend_file(asset)
    dl_url = blend_file.get("downloadUrl") or blend_file.get("fileThumbnailLarge")
    if not dl_url:
        sys.exit(f"No downloadUrl in file metadata: {blend_file}")

    base_dir = ASSET_TYPE_TO_DIR[asset_type]
    # Name the folder after the asset slug + short id
    slug = asset.get("slug") or asset_name.replace(" ", "-")
    folder_name = f"{slug}_{asset_base_id[:8]}"
    dest_dir = base_dir / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # BlenderKit returns a zip; filename comes from URL path
    url_name = Path(urlparse(dl_url).path).name or f"{asset_base_id}.zip"
    tmp_path = dest_dir / url_name
    print(f"Downloading {dl_url} → {tmp_path}")
    download(dl_url, tmp_path, token)

    blends = extract_if_zip(tmp_path, dest_dir, target_stem=folder_name)
    print(f"Done. {len(blends)} .blend file(s) in {dest_dir}:")
    for b in blends:
        print(f"  {b.relative_to(REPO_ROOT)}")
    if blends:
        spawn_background_inspect(blends[0])


if __name__ == "__main__":
    main()
