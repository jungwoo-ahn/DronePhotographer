"""The laptop's own way to the policy server, so nobody has to keep an `ssh -L` window open.

Forwards one local port to the same port on the server, over SSH with a saved password. This is
only a pipe: `PolicyClient` talks through it exactly as it would through `ssh -N -L`, and a dead
server looks to it like any other dead link.

The server's address and password live in Windows Credential Manager, under this user's account,
and never in this repository. Saving them also pins the server's host key, because from then on a
password goes out with nobody watching, and it must only ever go to that machine.
"""

from __future__ import annotations

import base64
import json
import select
import socket
import threading

CREDENTIAL_SERVICE = "DronePhotographer"
CREDENTIAL_NAME = "server"


def save_server(host: str, port: int, user: str, password: str) -> None:
    """Check the password works, then keep it, along with the key the server has now."""
    import keyring
    import paramiko

    transport = paramiko.Transport((host, port))
    try:
        transport.connect(username=user, password=password)
        key = transport.get_remote_server_key()
    finally:
        transport.close()
    keyring.set_password(CREDENTIAL_SERVICE, CREDENTIAL_NAME, json.dumps({
        "host": host, "port": port, "user": user, "password": password,
        "key_type": key.get_name(), "key": base64.b64encode(key.asbytes()).decode(),
    }))


def load_server() -> dict | None:
    """What `save_server` kept, or `None`, including when there is no credential store here."""
    try:
        import keyring
        saved = keyring.get_password(CREDENTIAL_SERVICE, CREDENTIAL_NAME)
    except Exception:
        return None
    return json.loads(saved) if saved else None


def _connect_ssh(server: dict):
    import paramiko

    key = paramiko.PKey.from_type_string(server["key_type"], base64.b64decode(server["key"]))
    transport = paramiko.Transport((server["host"], server["port"]))
    try:
        # Refuses before sending the password if the server's key is not the saved one.
        transport.connect(hostkey=key, username=server["user"], password=server["password"])
    except Exception:
        transport.close()
        raise
    transport.set_keepalive(15)
    return transport


class ServerTunnel:
    """Connects when the first request comes, and again whenever the connection has died."""

    def __init__(self, server: dict, port: int, connect=_connect_ssh):
        self.server = server
        self.port = int(port)
        self._connect = connect
        self._lock = threading.Lock()
        self._transport = None
        self._listener: socket.socket | None = None
        self.last_error: str | None = None

    def start(self) -> "ServerTunnel":
        """Raises `OSError` if the port is taken, which may be an `ssh -L` already doing this."""
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._listener.bind(("127.0.0.1", self.port))
            self._listener.listen()
        except OSError:
            self._listener.close()
            raise
        threading.Thread(target=self._accept, name="tunnel", daemon=True).start()
        return self

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.close()
        with self._lock:
            if self._transport is not None:
                self._transport.close()
                self._transport = None

    def _accept(self) -> None:
        while True:
            try:
                local, peer = self._listener.accept()
            except OSError:         # stopped
                return
            threading.Thread(target=self._serve, args=(local, peer), daemon=True).start()

    def _channel(self, peer):
        with self._lock:
            if self._transport is None or not self._transport.is_active():
                self._transport = self._connect(self.server)
            return self._transport.open_channel(
                "direct-tcpip", ("127.0.0.1", self.port), peer)

    def _serve(self, local: socket.socket, peer) -> None:
        channel = None
        try:
            channel = self._channel(peer)
            self.last_error = None
            while True:
                ready, _, _ = select.select([local, channel], [], [], 300)
                if not ready:
                    break
                if local in ready:
                    data = local.recv(65536)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in ready:
                    data = channel.recv(65536)
                    if not data:
                        break
                    local.sendall(data)
        except Exception as e:      # the client sees its connection close, and reports a lost link
            self.last_error = f"{type(e).__name__}: {e}"
        finally:
            local.close()
            if channel is not None:
                channel.close()
