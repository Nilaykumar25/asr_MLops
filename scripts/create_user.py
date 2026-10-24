"""Create API users.
  python -m scripts.create_user alice
  python -m scripts.create_user --count 50 --prefix load --keys-file loadtests/keys.txt
The plaintext key is shown ONCE; only its SHA-256 hash is stored."""
import argparse
import secrets

from sqlalchemy import select

from app.api.deps import hash_key
from app.config import get_settings
from app.db.models import User
from app.db.session import session_scope


def make_user(db, username: str, max_runs: int) -> str:
    if db.scalar(select(User).where(User.username == username)):
        raise SystemExit(f"user '{username}' already exists")
    key = secrets.token_urlsafe(32)
    db.add(User(username=username, api_key_hash=hash_key(key), max_concurrent_runs=max_runs))
    return key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("username", nargs="?")
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--prefix", default="load")
    p.add_argument("--keys-file")
    p.add_argument("--max-runs", type=int, default=get_settings().default_max_concurrent_runs)
    a = p.parse_args()

    if a.count:
        keys = []
        with session_scope() as db:
            for i in range(a.count):
                keys.append(make_user(db, f"{a.prefix}{i:03d}", a.max_runs))
        if a.keys_file:
            with open(a.keys_file, "w") as f:
                f.write("\n".join(keys) + "\n")
        print(f"created {a.count} users" + (f"; keys written to {a.keys_file}" if a.keys_file else ""))
    elif a.username:
        with session_scope() as db:
            key = make_user(db, a.username, a.max_runs)
        print(f"user: {a.username}\nAPI key (save it now, it is not stored): {key}")
    else:
        p.error("give a username or --count")


if __name__ == "__main__":
    main()
