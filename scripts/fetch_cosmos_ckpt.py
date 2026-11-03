"""Fetch one cosmos3 checkpoint from the training cluster over SFTP. Weights only.

    pip install paramiko
    export CKPT_SSH_PASSWORD=...          # or use --key for a private key
    python scripts/fetch_cosmos_ckpt.py \
        --host ssh-nipagpu.kakaocloud.com --port 32603 --user <you> \
        --run  /home/.../action_sft/camera_policy_nano_v6 \
        --iter iter_000045000 \
        --dest checkpoints/camera_policy_nano_v6/iter_000045000

Checkpoints are never committed -- `checkpoints/` is in `.gitignore` and a single
iteration is ~91 GB. This script is how the repo carries the checkpoint instead: the
recipe, not the bytes.

`--iter` names one iteration directory. Everything under it comes down *except*
`optim/`, which is another 56 GB of Adam state that only a resumed training run would
want. Pass `--include-optim` if you are resuming rather than serving.

After this, consolidate the FSDP shards into something loadable with
`scripts/consolidate_cosmos_ckpt.py`; see `docs/cosmos_checkpoint.md`.
"""

import argparse
import getpass
import os
import stat
import sys
import threading
import time

MB = 1e6
GB = 1e9


def connect(a, password):
    import paramiko

    t = paramiko.Transport((a.host, a.port))
    if a.key:
        t.connect(username=a.user, pkey=paramiko.Ed25519Key.from_private_key_file(a.key)
                  if "ed25519" in a.key.lower()
                  else paramiko.RSAKey.from_private_key_file(a.key))
    else:
        t.connect(username=a.user, password=password)
    import paramiko as _p
    return t, _p.SFTPClient.from_transport(t)


def walk(sftp, remote):
    """Every file under `remote`, as (path, size). Directories are recursed."""
    out = []
    for e in sftp.listdir_attr(remote):
        p = f"{remote}/{e.filename}"
        if stat.S_ISDIR(e.st_mode):
            out += walk(sftp, p)
        else:
            out.append((p, e.st_size))
    return out


def manifest(sftp, a):
    """What to fetch, as (remote, size, local-relative-path)."""
    files = []
    for f in ("config.yaml", "config.pkl", "job_env.yaml", "launch_info.yaml",
              "wandb_id.txt"):
        try:
            files.append((f"{a.run}/{f}", sftp.stat(f"{a.run}/{f}").st_size, f))
        except IOError:
            pass                                  # not every run writes every one

    base = f"{a.run}/checkpoints/{a.iter}"
    subs = ["KEEP", "model", "scheduler", "trainer"]
    if a.include_optim:
        subs.append("optim")
    for sub in subs:
        p = f"{base}/{sub}"
        try:
            e = sftp.stat(p)
        except IOError:
            continue
        if stat.S_ISDIR(e.st_mode):
            for rp, sz in walk(sftp, p):
                files.append((rp, sz, os.path.relpath(rp, base)))
        else:
            files.append((p, e.st_size, sub))
    return files


def is_complete(local, size):
    """Already fully here?

    `os.path.exists` first, deliberately. The obvious form of this test is
    `os.path.getsize(local) == size`, defaulting a missing file to 0 -- and for a
    *zero-byte* remote file that reads 0 == 0 and declares the file fetched before it
    has ever been created. An earlier version of this fetcher did exactly that and
    silently skipped seven empty `scheduler/__N_0.distcp` files while still reporting
    the right byte total, because the total only summed files that existed.
    """
    return os.path.exists(local) and os.path.getsize(local) == size


def fetch_one(a, password, remote, size, local, tag, lock, done):
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    if is_complete(local, size):
        with lock:
            done.append(tag)
            if a.verbose:
                print(f"  [{len(done)}] {tag}  cached", flush=True)
        return
    have = os.path.getsize(local) if os.path.exists(local) else 0
    if have > size:
        have = 0                                  # longer than the remote: start over
    if size == 0:
        open(local, "wb").close()                 # an empty file is still a file
        with lock:
            done.append(tag)
        return

    t, sftp = connect(a, password)
    try:
        sftp.get_channel().settimeout(a.timeout)
        t0 = time.time()
        with sftp.open(remote, "rb") as rf, open(local, "ab" if have else "wb") as lf:
            rf.prefetch(size)
            if have:
                rf.seek(have)
            while True:
                b = rf.read(8 << 20)
                if not b:
                    break
                lf.write(b)
        dt, mb = time.time() - t0, (size - have) / MB
        with lock:
            done.append(tag)
            print(f"  [{len(done)}] {tag}  {mb:,.0f} MB in {dt:,.0f}s "
                  f"({mb / max(dt, 1e-9):,.0f} MB/s)"
                  + (f"  (resumed at {have / MB:,.0f} MB)" if have else ""), flush=True)
    finally:
        sftp.close()
        t.close()


def check(files, dest):
    """Every file present at exactly the remote size, named individually if not.

    This is the only honest completion test: summing the bytes that arrived and
    comparing to the bytes expected passes even when a file is missing, as long as
    whatever is missing is empty.
    """
    bad = []
    for _, size, rel in files:
        local = os.path.join(dest, rel)
        if not os.path.exists(local):
            bad.append((rel, size, "missing"))
        elif os.path.getsize(local) != size:
            bad.append((rel, size, f"{os.path.getsize(local)} B, expected {size}"))
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--user", required=True)
    p.add_argument("--run", required=True, help="the run directory on the cluster")
    p.add_argument("--iter", required=True, help="e.g. iter_000045000")
    p.add_argument("--dest", required=True, help="where to put it locally")
    p.add_argument("--key", default=None, help="private key file (else password auth)")
    p.add_argument("--jobs", type=int, default=4, help="shards in flight at once")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--include-optim", action="store_true",
                   help="also fetch optim/ (+56 GB; only needed to resume training)")
    p.add_argument("--dry-run", action="store_true", help="print the manifest and stop")
    p.add_argument("--verify", action="store_true",
                   help="check what is already on disk against the cluster, then stop")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    password = None
    if not a.key:
        password = os.environ.get("CKPT_SSH_PASSWORD")
        if not password:
            # Never read this from a file in the tree: an earlier run of this kept the
            # password beside the script and it had to be shredded afterwards.
            password = getpass.getpass(f"password for {a.user}@{a.host}: ")

    t, sftp = connect(a, password)
    try:
        files = manifest(sftp, a)
    finally:
        sftp.close()
        t.close()
    if not files:
        raise SystemExit(f"nothing found under {a.run}/checkpoints/{a.iter}")

    total = sum(s for _, s, _ in files)
    print(f"{len(files)} files, {total / GB:,.2f} GB -> {a.dest}", flush=True)

    if a.dry_run:
        for _, size, rel in sorted(files, key=lambda f: -f[1]):
            print(f"  {size / MB:12,.1f} MB  {rel}")
        return

    if a.verify:
        bad = check(files, a.dest)
        for rel, _, why in bad:
            print(f"  BAD  {rel}: {why}")
        print(f"{len(files) - len(bad)}/{len(files)} files byte-exact against the cluster")
        sys.exit(1 if bad else 0)

    lock, done = threading.Lock(), []
    small = [f for f in files if f[1] <= 50 * MB]
    big = [f for f in files if f[1] > 50 * MB]

    for rp, sz, rel in small:
        fetch_one(a, password, rp, sz, os.path.join(a.dest, rel), rel, lock, done)
    print(f"{len(small)} small files done; {len(big)} shards, {a.jobs} at a time",
          flush=True)

    queue = list(big)
    qlock = threading.Lock()

    def worker():
        while True:
            with qlock:
                if not queue:
                    return
                rp, sz, rel = queue.pop(0)
            fetch_one(a, password, rp, sz, os.path.join(a.dest, rel), rel, lock, done)

    threads = [threading.Thread(target=worker) for _ in range(max(1, a.jobs))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    bad = check(files, a.dest)
    print()
    if bad:
        for rel, _, why in bad:
            print(f"  BAD  {rel}: {why}")
        raise SystemExit(f"{len(bad)} of {len(files)} files are wrong -- rerun to resume")
    print(f"all {len(files)} files byte-exact, {total / GB:,.2f} GB")
    print("next: python scripts/consolidate_cosmos_ckpt.py --src "
          f"{a.dest} --weights ema --dtype bf16")


if __name__ == "__main__":
    main()
