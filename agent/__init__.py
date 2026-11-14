import os
from pathlib import Path


def _load_env(path=Path(__file__).resolve().parent.parent / ".env"):
    """Read KEY=VALUE lines from the repo's gitignored .env without overriding the environment."""
    if path.is_file():
        for line in path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() and v.strip() and not k.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
