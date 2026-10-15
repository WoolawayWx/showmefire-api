"""Cross-process lock around the shared status.json read-modify-write.

maps.sh runs several maps/*.py scripts in parallel, and each one that
reports into status.json reads the whole file, sets its own key, and writes
it all back - without a lock, two scripts finishing together would drop each
other's update. Scripts call acquire() before reading status.json and
release() after writing it.
"""
import fcntl

_lock_file = None


def acquire(status_path):
    global _lock_file
    _lock_file = open(f"{status_path}.lock", "w")
    fcntl.flock(_lock_file, fcntl.LOCK_EX)


def release():
    global _lock_file
    if _lock_file is not None:
        fcntl.flock(_lock_file, fcntl.LOCK_UN)
        _lock_file.close()
        _lock_file = None
