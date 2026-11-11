"""Save the policy server's login once, so the pilot can connect to it by itself.

    python scripts/save_server_login.py --host <server> --port <ssh port> --user <user>

It asks for the password, checks that it works, and keeps it in Windows Credential Manager. Run it
again after changing the password, or if the server's host key has changed for a good reason.
"""

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drone_control.tunnel import save_server


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--user", required=True)
    a = p.parse_args()
    save_server(a.host, a.port, a.user, getpass.getpass(f"password for {a.user}@{a.host}: "))
    print("saved")


if __name__ == "__main__":
    main()
