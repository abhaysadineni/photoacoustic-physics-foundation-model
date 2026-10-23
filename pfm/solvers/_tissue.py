"""Read-only access to params/tissue_params.json, by parameter id.

Every tissue property and physical constant used by the tissue solvers (light, heat, sound,
chain) is read through this module at import time. A value that is null, or an entry whose
status is UNVERIFIED, raises MissingParameter instead of being replaced by a guess.

Unit conversions used by the solvers are definitions, not tissue data:
    1 cm^-1 = 100 m^-1 (CM_TO_M), 1 MHz = 1e6 Hz, 1 Np = 20 / ln(10) dB (the JSON
    "conventions.attenuation" entry states the same factor).
"""
from __future__ import annotations

import json
import math
import re

from pfm.common import ROOT

PATH = ROOT / "params" / "tissue_params.json"
_DATA = json.loads(PATH.read_text())
_BY_ID = {p["id"]: p for p in _DATA["parameters"]}

CM_TO_M = 100.0                       # 1 cm^-1 = 100 m^-1 (unit definition)
DB_TO_NP = math.log(10.0) / 20.0      # 1 dB = ln(10)/20 Np (definition; JSON conventions)


class MissingParameter(RuntimeError):
    """A needed value is absent, null or UNVERIFIED in tissue_params.json."""


def entry(pid: str) -> dict:
    if pid not in _BY_ID:
        raise MissingParameter(f"{pid}: not in {PATH.name}")
    e = _BY_ID[pid]
    if str(e.get("status", "")).upper().startswith("UNVERIFIED"):
        raise MissingParameter(f"{pid}: status UNVERIFIED, value {e.get('value')!r}")
    return e


def get(pid: str, *path, field: str = "value"):
    """entry(pid)[field][path...]; raises if any level is missing or null."""
    v = entry(pid).get(field)
    where = f"{pid}.{field}"
    for k in path:
        if v is None:
            break
        if isinstance(v, list):
            # list of dicts keyed by a 'tissue' name
            hits = [x for x in v if isinstance(x, dict) and x.get("tissue") == k]
            if len(hits) != 1:
                raise MissingParameter(f"{where}: no unique row '{k}'")
            v = hits[0]
        else:
            if k not in v:
                raise MissingParameter(f"{where}: no key '{k}'")
            v = v[k]
        where += f".{k}"
    if v is None:
        raise MissingParameter(f"{where}: null")
    return v


def leading_float(s) -> float:
    """Leading number of a string such as '0.81 +/- 0.05 (n=6)'."""
    if isinstance(s, (int, float)):
        return float(s)
    m = re.match(r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", str(s))
    if not m:
        raise MissingParameter(f"cannot parse a number from {s!r}")
    return float(m.group(1))


def reference_wavelength_nm(pid: str, field: str) -> float:
    """The reference wavelength written inside a sourced power law, e.g. '(lambda/500 nm)'."""
    s = entry(pid)[field]
    m = re.search(r"lambda\s*/\s*(\d+(?:\.\d+)?)\s*nm", s)
    if not m:
        raise MissingParameter(f"{pid}.{field}: no '(lambda/<x> nm)' in {s!r}")
    return float(m.group(1))


def first_key(pid: str, prefix: str, field: str = "value_SI"):
    """Value of the first key starting with prefix in a dict-valued field."""
    d = get(pid, field=field)
    for k, v in d.items():
        if k.startswith(prefix):
            if v is None:
                raise MissingParameter(f"{pid}.{field}.{k}: null")
            return float(v), k
    raise MissingParameter(f"{pid}.{field}: no key starting with {prefix}")
