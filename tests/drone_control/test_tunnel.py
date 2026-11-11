"""The pipe to the server, with the SSH connection replaced by an echoing socket."""

import socket
import threading

import pytest

from src.drone_control import tunnel as tunnel_module
from src.drone_control.tunnel import ServerTunnel, load_server


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EchoTransport:
    """Each channel is a socket whose other end sends back what it gets."""

    def __init__(self):
        self.opened = []
        self.active = True

    def is_active(self):
        return self.active

    def close(self):
        self.active = False

    def open_channel(self, kind, destination, peer):
        self.opened.append((kind, destination))
        ours, theirs = socket.socketpair()

        def echo():
            while data := theirs.recv(1024):
                theirs.sendall(data)
            theirs.close()

        threading.Thread(target=echo, daemon=True).start()
        return ours


def ask(port, message=b"hello"):
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as c:
        c.settimeout(2.0)
        c.sendall(message)
        return c.recv(1024)


def test_bytes_go_through_to_the_same_port_on_the_server():
    transport = EchoTransport()
    port = free_port()
    t = ServerTunnel({}, port, connect=lambda s: transport).start()
    assert ask(port) == b"hello"
    assert transport.opened == [("direct-tcpip", ("127.0.0.1", port))]
    t.stop()
    assert not transport.active


def test_it_connects_only_once_while_the_connection_lives():
    connects = []

    def connect(server):
        connects.append(server)
        return EchoTransport()

    port = free_port()
    t = ServerTunnel({}, port, connect=connect).start()
    assert ask(port) == b"hello" and ask(port) == b"hello"
    assert len(connects) == 1
    t.stop()


def test_it_connects_again_after_the_connection_dies():
    transports = []

    def connect(server):
        transports.append(EchoTransport())
        return transports[-1]

    port = free_port()
    t = ServerTunnel({}, port, connect=connect).start()
    assert ask(port) == b"hello"
    transports[-1].active = False
    assert ask(port) == b"hello"
    assert len(transports) == 2
    t.stop()


def test_an_unreachable_server_closes_the_request_and_says_why():
    def connect(server):
        raise OSError("no route")

    port = free_port()
    t = ServerTunnel({}, port, connect=connect).start()
    try:
        assert ask(port) == b""
    except ConnectionError:     # how Windows reports the same close
        pass
    assert "no route" in t.last_error
    t.stop()


def test_a_taken_port_is_reported_at_once():
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        with pytest.raises(OSError):
            ServerTunnel({}, holder.getsockname()[1]).start()


def test_nothing_saved_is_none(monkeypatch):
    keyring = pytest.importorskip("keyring")
    monkeypatch.setattr(keyring, "get_password", lambda service, name: None)
    assert load_server() is None


def test_no_credential_store_is_none_too(monkeypatch):
    keyring = pytest.importorskip("keyring")

    def broken(service, name):
        raise RuntimeError("no backend")

    monkeypatch.setattr(keyring, "get_password", broken)
    assert load_server() is None


def test_a_changed_host_key_is_refused_before_the_password_goes_out():
    paramiko = pytest.importorskip("paramiko")
    seen = {}

    class Transport:
        def __init__(self, address):
            pass

        def connect(self, hostkey=None, username=None, password=None):
            seen["hostkey"] = hostkey
            raise paramiko.SSHException("Bad host key from server")

        def close(self):
            seen["closed"] = True

    key = paramiko.RSAKey.generate(1024)
    server = {"host": "h", "port": 1, "user": "u", "password": "p", "key_type": key.get_name(),
              "key": __import__("base64").b64encode(key.asbytes()).decode()}
    original = paramiko.Transport
    paramiko.Transport = Transport
    try:
        with pytest.raises(paramiko.SSHException):
            tunnel_module._connect_ssh(server)
    finally:
        paramiko.Transport = original
    assert seen["hostkey"] == key and seen["closed"]
