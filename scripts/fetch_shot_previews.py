"""Copy each evaluation shot's goal render to the laptop, so the panel can show the photo.

    python scripts/fetch_shot_previews.py            # into outputs/shot_previews

Every shot in the panel's list comes from an evaluation episode, and each episode's profile
was measured from one rendered frame: its goal. That frame is the clearest picture of what
the shot asks for. The renders stay on the server where training left them; this only reads
them, over the login `save_server_login.py` keeps, and writes small copies here. Run it once;
it skips what it already has.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drone_control.shots import EPISODES, preview_name  # noqa: E402
from src.drone_control.tunnel import _connect_ssh, load_server  # noqa: E402


def shrink(jpeg: bytes, width: int) -> bytes:
    """The same picture, `width` pixels wide, still a JPEG."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not an image")
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, out = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("could not encode")
    return out.tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/shot_previews")
    p.add_argument("--episodes", default=str(EPISODES))
    p.add_argument("--width", type=int, default=512)
    a = p.parse_args()

    with open(a.episodes, encoding="utf-8") as f:
        episodes = json.load(f)["episodes"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [(i, e) for i, e in enumerate(episodes) if not (out / preview_name(i)).exists()]
    if not todo:
        print(f"all {len(episodes)} already in {out}")
        return
    server = load_server()
    if server is None:
        raise SystemExit("no saved server login; run scripts/save_server_login.py first")

    import paramiko
    transport = _connect_ssh(server)
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        for i, e in todo:
            with sftp.open(e["goal_image"], "rb") as f:
                (out / preview_name(i)).write_bytes(shrink(f.read(), a.width))
            print(f"{preview_name(i)}  {e['sector']:11s}  {Path(e['goal_image']).name}")
    finally:
        sftp.close()
        transport.close()
    print(f"{len(todo)} copied into {out.resolve()}")


if __name__ == "__main__":
    main()
