from re import Match
from typing import Any

class JSONDecoder:
    def __init__(self, **kwargs: Any) -> None: ...
    def decode(self, s: str, _w: Match[str], _PY3: bool): ...
    def raw_decode(self, s: str, idx: int, _w: Match[str], _PY3: bool): ...
