"""BSIM4Model — model card data with defaults from official C source.

Public API:
    m = BSIM4Model(vth0=0.5, u0=0.045)            # explicit overrides
    m = BSIM4Model.from_spice(text)                # parse .model card
    m["vth0"]    →  0.5
    m.is_given("vth0")  →  True
    m.is_given("k1")    →  False  (default used)

The dict ALIASES handles legacy SPICE names (e.g. 'vtho' for 'vth0').
"""
from __future__ import annotations
import re
from typing import Any

from ._model_card_data import PARAMS_META, ALIASES


def _resolve_default(default: Any, type_n: int, scratch: dict[str, Any]) -> Any:
    if isinstance(default, tuple):
        kind = default[0]
        if kind == "nmos_pmos":
            return default[1] if type_n == 1 else default[2]
        if kind == "mobmod":
            return default[1]
        if kind == "ref":
            return scratch.get(default[1], 0.0)
    return default


# SPICE engineering-suffix table (n→1e-9, u→1e-6, etc.).
SPICE_SUFFIX = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6,
    "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12,
}


def parse_spice_value(s: str) -> float | None:
    """Parse a SPICE-style numeric literal: '4n', '1.35e5', '-0.0465', '2u', etc."""
    s = s.strip()
    # Plain float / scientific notation
    try:
        return float(s)
    except ValueError:
        pass
    # Engineering suffix: digits[.digits][e±dd]<suffix>
    m = re.match(r"^([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*([a-zA-Z]+)$", s)
    if m:
        try:
            base = float(m.group(1))
        except ValueError:
            return None
        suf = m.group(2).lower()
        # Check 3-letter then 1-letter
        for k in ("meg",):
            if suf == k:
                return base * SPICE_SUFFIX[k]
        if suf and suf[0] in SPICE_SUFFIX:
            return base * SPICE_SUFFIX[suf[0]]
    return None


class BSIM4Model:
    """Holds ~880 BSIM4 v4.8.3 model parameters with C-faithful defaults."""

    __slots__ = ("_values", "_given")

    def __init__(self, **kwargs):
        type_arg = kwargs.pop("type", 1)
        if isinstance(type_arg, str):
            type_arg = 1 if type_arg.upper() == "NMOS" else -1
        self._values: dict[str, Any] = {"type": type_arg}
        self._given: set[str] = set()

        # Pass 1: scalar + nmos_pmos + mobmod defaults
        for name, info in PARAMS_META.items():
            d = info["default"]
            if isinstance(d, tuple) and d[0] == "ref":
                continue
            self._values[name] = _resolve_default(d, type_arg, self._values)

        # Pass 2: refs (after scalar defaults populated)
        for name, info in PARAMS_META.items():
            d = info["default"]
            if isinstance(d, tuple) and d[0] == "ref":
                self._values[name] = self._values.get(d[1], 0.0)

        # Pass 3: user overrides
        for k, v in kwargs.items():
            self.set(k, v)

    def _canonical(self, name: str) -> str:
        n = name.lower()
        return ALIASES.get(n, n)

    def __getitem__(self, name: str) -> Any:
        return self._values[self._canonical(name)]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        c = self._canonical(name)
        if c in self._values:
            return self._values[c]
        raise AttributeError(name)

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(self._canonical(name), default)

    def set(self, name: str, value: Any) -> None:
        c = self._canonical(name)
        if c not in PARAMS_META:
            raise KeyError(f"unknown BSIM4 parameter: {name!r}")
        info = PARAMS_META[c]
        if info["py_type"] == "float":
            coerced = float(value)
        elif info["py_type"] == "int":
            coerced = int(value)
        elif info["py_type"] == "bool":
            coerced = bool(int(value)) if isinstance(value, str) else bool(value)
        else:
            coerced = value
        self._values[c] = coerced
        self._given.add(c)

    def is_given(self, name: str) -> bool:
        return self._canonical(name) in self._given

    @property
    def given(self) -> frozenset[str]:
        return frozenset(self._given)

    def asdict(self, only_given: bool = False) -> dict[str, Any]:
        if only_given:
            return {k: self._values[k] for k in self._given}
        return dict(self._values)

    @classmethod
    def from_spice(cls, text: str, *, params: dict[str, float] | None = None,
                    model_type: str = "nmos") -> "BSIM4Model":
        """Parse a SPICE .model card. Handles continuation '+', .param, suffixes.

        If the file has multiple .model cards (e.g. NMOS and PMOS), pick the
        one matching `model_type` (default 'nmos').
        """
        params = {k.lower(): v for k, v in (params or {}).items()}
        # Pre-extract .param definitions (these are file-wide).
        # ngspice accepts BOTH `.param name=value` and `.param name value`.
        for line in text.splitlines():
            mp = re.match(r"\s*\.param\s+(\w+)\s*[=\s]\s*(\S+)", line, re.IGNORECASE)
            if mp:
                v = parse_spice_value(mp.group(2))
                if v is not None:
                    params[mp.group(1).lower()] = v

        # Slice to just the requested .model block: from its .model line to
        # either the next .model or end-of-file. Continuation '+' lines belong.
        model_re = re.compile(r"^\s*\.model\s+\S+\s+(\w+)", re.MULTILINE | re.IGNORECASE)
        starts = list(model_re.finditer(text))
        if starts:
            chosen = None
            for i, m in enumerate(starts):
                if m.group(1).lower() == model_type.lower():
                    chosen = m
                    end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
                    text = text[m.start():end]
                    break
            if chosen is None:
                # No matching model — use the first one anyway, but warn.
                m = starts[0]
                end = starts[1].start() if len(starts) > 1 else len(text)
                text = text[m.start():end]

        # Detect type from .model line (post-slice).
        type_kw = 1
        m_model = re.search(r"\.model\s+\S+\s+(nmos|pmos)", text, re.IGNORECASE)
        if m_model:
            type_kw = 1 if m_model.group(1).lower() == "nmos" else -1

        m = cls(type=type_kw)

        # Strip comments + .param + .end
        cleaned: list[str] = []
        for raw in text.splitlines():
            ln = raw.split("$")[0]
            if re.match(r"\s*\*", ln):
                continue
            if re.match(r"\s*\.param\s+", ln, re.IGNORECASE):
                continue
            if re.match(r"\s*\.end", ln, re.IGNORECASE):
                continue
            cleaned.append(ln)

        # Join '+' continuation lines.
        merged: list[str] = []
        for line in cleaned:
            if line.lstrip().startswith("+"):
                if merged:
                    merged[-1] = merged[-1] + " " + line.lstrip()[1:]
                else:
                    merged.append(line.lstrip()[1:])
            else:
                merged.append(line)
        flat = " ".join(merged)

        # Strip the .model header so its tokens aren't mistaken for params.
        flat = re.sub(r"\.model\s+\S+\s+\S+", " ", flat, flags=re.IGNORECASE)

        # Extract `name=value` pairs. Value may be:
        #   - SPICE numeric: [+-]?digits[.digits][e±d+][suffix-letters]
        #   - .param identifier reference
        # We match either; parse_spice_value disambiguates.
        VAL = r"(?:[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[a-zA-Z]*|[A-Za-z_]\w*)"
        for match in re.finditer(
            rf"(\w+)\s*=\s*({VAL})", flat
        ):
            name = match.group(1).lower()
            val = match.group(2).strip()
            if name in ("level", "version", "type"):
                continue
            canon = ALIASES.get(name, name)
            if canon not in PARAMS_META:
                continue
            v = parse_spice_value(val)
            if v is None:
                # Try .param substitution
                v = params.get(val.lower())
            if v is None:
                continue
            try:
                m.set(name, v)
            except (KeyError, ValueError):
                pass
        return m

    def __repr__(self) -> str:
        n = len(self._given)
        type_str = "NMOS" if self._values.get("type", 1) == 1 else "PMOS"
        return f"<BSIM4Model {type_str} given={n} of {len(PARAMS_META)}>"
