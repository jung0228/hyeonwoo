#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MacBook to Raspberry Pi Real-Time Auto-Sync Daemon
Monitors data/notes and data/knowledge.json for changes, and instantly rsyncs to Raspberry Pi (192.168.45.119).
"""

import os
import sys
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCH_PATHS = [
    os.path.join(BASE_DIR, "data"),
    os.path.join(BASE_DIR, "app.js"),
    os.path.join(BASE_DIR, "index.html"),
    os.path.join(BASE_DIR, "style.css")
]

def get_mtime_snapshot():
    snapshot = {}
    for p in WATCH_PATHS:
        if os.path.isfile(p):
            try:
                snapshot[p] = os.path.getmtime(p)
            except OSError:
                pass
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    fpath = os.path.join(root, f)
                    try:
                        snapshot[fpath] = os.path.getmtime(fpath)
                    except OSError:
                        pass
    return snapshot

def sync_to_pi():
    cmd = (
        "sshpass -p '1234' rsync -avz --delete "
        "--exclude '.git' --exclude 'node_modules' "
        f"{BASE_DIR}/ pi@192.168.45.119:~/hyeonwoo-web/"
    )
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            print(f"⚡ [{time.strftime('%H:%M:%S')}] MacBook ➡️ Raspberry Pi Auto-Sync Successful!")
        else:
            print(f"⚠️ Sync Error: {res.stderr.strip()}")
    except Exception as e:
        print(f"❌ Sync Exception: {e}")

def main():
    print("📡 MacBook ➡️ Raspberry Pi Real-Time Auto-Sync Daemon Started!")
    last_snapshot = get_mtime_snapshot()
    
    while True:
        try:
            time.sleep(2)
            current_snapshot = get_mtime_snapshot()
            if current_snapshot != last_snapshot:
                print("📝 File change detected on MacBook! Syncing to Raspberry Pi...")
                sync_to_pi()
                last_snapshot = current_snapshot
        except KeyboardInterrupt:
            print("\nStopping Auto-Sync Daemon.")
            sys.exit(0)
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(3)

if __name__ == "__main__":
    main()
