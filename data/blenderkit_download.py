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
import json
import os
import queue
import readline  # noqa: F401  -- makes input() use readline so get_line_buffer() works
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from rich.console import Console

_console = Console(highlight=False, markup=True, file=sys.stdout, soft_wrap=True)
_USE_ANSI = sys.stdout.isatty()


class _UI:
    """Footer-style UI: a status line + prompt pinned at the bottom, log output above.

    Any call to `log()` from any thread clears the two footer lines, prints the
    message, and redraws the footer — preserving the user's in-progress input
    via `readline.get_line_buffer()`.
    """

    PROMPT = "\x1b[1;36m❯\x1b[0m "

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.status = "idle"
        self.active = False

    def start(self) -> None:
        with self.lock:
            self.active = True
            if _USE_ANSI:
                sys.stdout.write("\n")  # reserve the status line above the prompt
                self._draw_footer()

    def stop(self) -> None:
        with self.lock:
            if self.active and _USE_ANSI:
                self._clear_footer()
                sys.stdout.write("\n")
                sys.stdout.flush()
            self.active = False

    def _buf(self) -> str:
        try:
            import readline as _rl
            return _rl.get_line_buffer()
        except Exception:
            return ""

    def _draw_footer(self) -> None:
        status = self.status or "idle"
        sys.stdout.write(f"\x1b[2K\x1b[2m⧗ {status}\x1b[0m\n")
        sys.stdout.write(f"\x1b[2K{self.PROMPT}{self._buf()}")
        sys.stdout.flush()

    def _clear_footer(self) -> None:
        # Cursor is somewhere on the prompt line; clear it, go up, clear status line.
        sys.stdout.write("\r\x1b[2K\x1b[1A\x1b[2K")
        sys.stdout.flush()

    def log(self, msg: str, style: str = "") -> None:
        with self.lock:
            if not self.active or not _USE_ANSI:
                _console.print(msg, style=style)
                return
            self._clear_footer()
            _console.print(msg, style=style)
            self._draw_footer()

    def set_status(self, msg: str) -> None:
        with self.lock:
            self.status = msg
            if not self.active or not _USE_ANSI:
                return
            self._clear_footer()
            self._draw_footer()

    def after_input(self) -> None:
        """Call once input() returns — the Enter keystroke scrolled the footer up, so redraw it."""
        with self.lock:
            if self.active and _USE_ANSI:
                self._draw_footer()


_ui = _UI()

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
            last = 0.0
            started = time.time()
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                total += len(chunk)
                now = time.time()
                if now - last >= 0.3:
                    mbps = (total / 1e6) / max(now - started, 1e-6)
                    _ui.set_status(
                        f"downloading {dest.name} — {total/1e6:.1f} MB ({mbps:.1f} MB/s)"
                    )
                    last = now
    _ui.log(f"[green]✓[/green] downloaded [bold]{dest.name}[/bold] ({total/1e6:.1f} MB)")


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


def run_inspect_sync(blend_path: Path, blender_bin: str = BLENDER_BIN_DEFAULT) -> dict | None:
    """Run blender_inspect.py on a .blend, write inspection.json, print verdict.

    Runs synchronously — returns the parsed report dict, or None on failure.
    """
    report_path = blend_path.parent / "inspection.json"
    _ui.set_status(f"inspecting {blend_path.name}")
    result = subprocess.run(
        [blender_bin, "--background", str(blend_path), "--python", str(INSPECT_SCRIPT)],
        capture_output=True, text=True,
    )
    verdict_line = None
    for line in result.stdout.splitlines():
        if line.startswith("###INSPECT### "):
            verdict_line = line[len("###INSPECT### "):]
            break
    if verdict_line is None:
        _ui.log(f"[yellow]⚠[/yellow] inspect [bold]{blend_path.name}[/bold]: no ###INSPECT### line in Blender output")
        return None
    report_path.write_text(verdict_line)
    try:
        report = json.loads(verdict_line)
    except Exception as e:
        _ui.log(f"[yellow]⚠[/yellow] inspect [bold]{blend_path.name}[/bold]: could not parse verdict: {e}")
        return None
    signals = report.get("signals") or []
    verdict = report.get("verdict", "?")
    color = "yellow" if signals else ("green" if verdict in {"ok", "likely_ok"} else "cyan")
    sig = f" [yellow]signals={signals}[/yellow]" if signals else ""
    _ui.log(
        f"[{color}]◉[/{color}] inspect [bold]{blend_path.name}[/bold] → "
        f"[{color}]{verdict}[/{color}]{sig} "
        f"[dim](meshes={report.get('mesh_count')}, polys={report.get('total_polys')})[/dim]"
    )
    return report


def download_one(asset_base_id: str, asset_type: str, token: str, force: bool = False) -> None:
    """Download one asset and inspect it. Prints progress + verdict.

    If a folder matching `*_<asset_base_id[:8]>` already exists under the
    target base_dir and contains a .blend, the download is skipped unless
    `force=True`.
    """
    if asset_type not in ASSET_TYPE_TO_DIR:
        _ui.log(f"[red]✗[/red] unknown asset_type: {asset_type}. "
                f"Pick one of: {list(ASSET_TYPE_TO_DIR)}")
        return

    # Duplicate check: any existing <slug>_<id8>/ with a .blend inside?
    base_dir = ASSET_TYPE_TO_DIR[asset_type]
    id8 = asset_base_id[:8]
    if not force:
        for d in base_dir.glob(f"*_{id8}"):
            if d.is_dir() and any(d.glob("*.blend")):
                _ui.log(f"[dim]· already downloaded: {d.relative_to(REPO_ROOT)} "
                        f"(pass force:true to re-download)[/dim]")
                return

    _ui.set_status(f"searching {asset_type}:{id8}")
    try:
        asset = search_asset(asset_base_id, asset_type, token)
    except SystemExit as e:
        _ui.log(f"[red]✗[/red] {e}")
        return
    asset_name = asset.get("name", asset_base_id)
    _ui.log(f"[blue]◆[/blue] found [bold]{asset_name}[/bold] [dim]({asset_type}:{id8})[/dim]")

    blend_file = pick_blend_file(asset)
    dl_url = blend_file.get("downloadUrl") or blend_file.get("fileThumbnailLarge")
    if not dl_url:
        _ui.log(f"[red]✗[/red] no downloadUrl in file metadata: {blend_file}")
        return

    slug = asset.get("slug") or asset_name.replace(" ", "-")
    folder_name = f"{slug}_{id8}"
    dest_dir = base_dir / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    url_name = Path(urlparse(dl_url).path).name or f"{asset_base_id}.zip"
    tmp_path = dest_dir / url_name
    download(dl_url, tmp_path, token)

    blends = extract_if_zip(tmp_path, dest_dir, target_stem=folder_name)
    _ui.log(f"[dim]→ {len(blends)} .blend file(s) in "
            f"{dest_dir.relative_to(REPO_ROOT)}[/dim]")
    if blends:
        run_inspect_sync(blends[0])


def _worker_loop(
    q: "queue.Queue[tuple[str, str, bool] | None]",
    token: str,
    state: dict,
) -> None:
    """Background worker that pulls (aid, atype, force) tuples and downloads them.

    A sentinel `None` tells the worker to exit.
    """
    def _idle_status() -> str:
        n = q.qsize()
        return f"{n} pending" if n else "idle"

    while True:
        item = q.get()
        try:
            if item is None:
                return
            aid, atype, force = item
            state["current"] = f"{atype}:{aid[:8]}"
            try:
                download_one(aid, atype, token, force=force)
            except Exception as e:
                _ui.log(f"[red]✗[/red] worker error on {atype}:{aid[:8]}: {e}")
            state["current"] = None
            _ui.set_status(_idle_status())
        finally:
            q.task_done()


def interactive_loop(token: str) -> None:
    """Read `asset_base_id:<uuid> asset_type:<type>` lines from stdin.

    Each line is enqueued and processed by a background worker so you can
    keep pasting while downloads/inspections run. Ctrl-D waits for the
    queue to drain, then exits.
    """
    q: "queue.Queue[tuple[str, str, bool] | None]" = queue.Queue()
    state: dict = {"current": None}
    worker = threading.Thread(target=_worker_loop, args=(q, token, state), daemon=True)
    worker.start()

    _ui.start()
    _ui.log("[bold cyan]BlenderKit interactive queue[/bold cyan] — paste requests; they run in the background.")
    _ui.log("[dim]format : asset_base_id:<uuid> asset_type:<scene|model|material|hdr|brush>[/dim]")
    _ui.log("[dim]cmds   : 'status' · 'quit' / 'exit' / Ctrl-D (drain and exit)[/dim]")
    _ui.set_status("idle")

    def drain_and_exit() -> None:
        pending = q.qsize()
        if pending or state["current"]:
            _ui.log(
                f"[yellow]⏳ draining[/yellow] {pending} queued + "
                f"{'1' if state['current'] else '0'} in-flight — please wait..."
            )
        q.put(None)
        worker.join()
        _ui.stop()

    while True:
        try:
            line = input().strip()  # prompt drawn by _ui footer
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            drain_and_exit()
            return
        _ui.after_input()
        if not line:
            continue
        if line.lower() in {"quit", "exit", ":q"}:
            drain_and_exit()
            return
        if line.lower() == "status":
            cur = state["current"] or "(idle)"
            _ui.log(f"[cyan]●[/cyan] queue → in-flight: [bold]{cur}[/bold] | pending: {q.qsize()}")
            continue
        try:
            kv = parse_kv_args(line.split())
        except SystemExit as e:
            _ui.log(f"[red]✗[/red] {e}"); continue
        aid = kv.get("asset_base_id")
        atype = kv.get("asset_type")
        if not aid or not atype:
            _ui.log("[yellow]need both asset_base_id:<uuid> and asset_type:<type>[/yellow]")
            continue
        force = kv.get("force", "").lower() in {"1", "true", "yes"}
        q.put((aid, atype, force))
        _ui.log(f"[blue]＋[/blue] queued [bold]{atype}:{aid[:8]}[/bold] "
                f"[dim](pending: {q.qsize()})[/dim]")
        if state["current"] is None:
            _ui.set_status(f"{q.qsize()} pending")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download BlenderKit assets and auto-inspect them.",
    )
    ap.add_argument("--keep-open", action="store_true",
                    help="Interactive mode: keep prompting for asset IDs until Ctrl-D.")
    ap.add_argument("kv", nargs="*",
                    help="key:value args, e.g. asset_base_id:<uuid> asset_type:scene")
    args = ap.parse_args()

    token = os.environ.get("BLENDERKIT_API_KEY", "")
    if not token:
        token_file = Path(__file__).parent / ".blenderkit_token"
        if token_file.is_file():
            token = token_file.read_text().strip()
    if not token:
        print("WARN: no API key found — set BLENDERKIT_API_KEY or write "
              "one to data/.blenderkit_token", file=sys.stderr)

    if args.keep_open:
        interactive_loop(token)
        return

    if not args.kv:
        ap.print_help()
        sys.exit(1)
    kv = parse_kv_args(args.kv)
    aid = kv.get("asset_base_id") or sys.exit("asset_base_id:<uuid> required")
    atype = kv.get("asset_type") or sys.exit("asset_type:<type> required")
    force = kv.get("force", "").lower() in {"1", "true", "yes"}
    download_one(aid, atype, token, force=force)


if __name__ == "__main__":
    main()
