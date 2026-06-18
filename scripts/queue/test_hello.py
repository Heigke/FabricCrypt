#!/usr/bin/env python3
"""Tiny smoke-test script for the queue framework.

Prints 'hello from $HOSTNAME', sleeps 5s, prints any args. Exits 0.
"""
import argparse
import os
import socket
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="smoke")
    p.add_argument("--sleep", type=float, default=5.0)
    args = p.parse_args()
    print(f"hello from {socket.gethostname()} tag={args.tag} pid={os.getpid()}", flush=True)
    print(f"argv={sys.argv}", flush=True)
    print(f"env HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION')}", flush=True)
    time.sleep(args.sleep)
    print(f"done {socket.gethostname()} tag={args.tag}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
