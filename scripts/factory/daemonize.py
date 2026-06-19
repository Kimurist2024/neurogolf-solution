#!/usr/bin/env python3
"""Launch a command as a detached double-fork daemon."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _redirect(stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    devnull = os.open(os.devnull, os.O_RDONLY)
    out = os.open(str(stdout_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    err = os.open(str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(devnull, 0)
    os.dup2(out, 1)
    os.dup2(err, 2)
    for fd in (devnull, out, err):
        try:
            os.close(fd)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")
    return args


def main() -> int:
    args = parse_args()
    read_fd, write_fd = os.pipe()

    first = os.fork()
    if first > 0:
        os.close(write_fd)
        with os.fdopen(read_fd, "rb", closefd=True) as pipe:
            payload = pipe.read().decode("ascii", errors="replace").strip()
        try:
            os.waitpid(first, 0)
        except ChildProcessError:
            pass
        if payload:
            print(payload)
            return 0
        print("daemonize failed: child did not report pid", file=sys.stderr)
        return 1

    os.close(read_fd)
    os.setsid()
    second = os.fork()
    if second > 0:
        os._exit(0)

    try:
        args.cwd.mkdir(parents=True, exist_ok=True)
        os.chdir(args.cwd)
        pid = os.getpid()
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(f"{pid}\n", encoding="utf-8")
        os.write(write_fd, f"{pid}\n".encode("ascii"))
        os.close(write_fd)
        _redirect(args.stdout, args.stderr)
        os.execvp(args.command[0], args.command)
    except Exception as exc:  # noqa: BLE001
        try:
            os.write(write_fd, f"ERROR {exc!r}\n".encode("utf-8", errors="replace"))
        except OSError:
            pass
        os._exit(127)


if __name__ == "__main__":
    raise SystemExit(main())
