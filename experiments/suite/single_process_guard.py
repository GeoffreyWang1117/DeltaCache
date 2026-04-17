"""Single-process guard for suite.run_all — prevents accidental double-launch.

Add this near the top of suite/run_all.py to ensure only one experiment runs
at a time. Uses a lock file + PID check to prevent GPU memory contention
on rented servers (each concurrent instance wastes $$).

Usage: just import this module. If another instance is running, exits immediately.
"""
import os
import sys
import time
import fcntl
import atexit

LOCK_PATH = "/tmp/deltacache_suite.lock"


def acquire_single_instance_lock():
    """Acquire exclusive lock, else exit. Releases on process exit."""
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError):
        # Lock held by another process
        try:
            with open(LOCK_PATH) as f:
                other_pid = f.read().strip()
            # Check if that PID is still alive
            try:
                os.kill(int(other_pid), 0)
                alive = True
            except (ValueError, ProcessLookupError, PermissionError):
                alive = False
        except Exception:
            alive = True

        if alive:
            print(f"ERROR: Another suite.run_all instance is running (PID {other_pid}).",
                  file=sys.stderr)
            print(f"Multiple instances would contend for GPU memory and waste compute.",
                  file=sys.stderr)
            print(f"If this is incorrect, remove {LOCK_PATH} manually.",
                  file=sys.stderr)
            sys.exit(2)
        else:
            # Stale lock — remove and retry
            os.remove(LOCK_PATH)
            return acquire_single_instance_lock()

    # Lock acquired — write our PID
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())

    def _release():
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            os.remove(LOCK_PATH)
        except Exception:
            pass
    atexit.register(_release)
