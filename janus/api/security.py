import re
from typing import Optional

ALLOWED_ORIGIN_REGEX = (
    r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|"
    r"10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$"
)
ALLOWED_ORIGIN_PATTERN = re.compile(ALLOWED_ORIGIN_REGEX)


def is_allowed_origin(origin: Optional[str]) -> bool:
    """Allows non-browser clients without an Origin header and local/RFC1918 browser origins."""
    if origin is None:
        return True
    return bool(ALLOWED_ORIGIN_PATTERN.fullmatch(origin.strip()))
