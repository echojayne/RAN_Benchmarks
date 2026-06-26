"""Minimal evaluation helpers required by the channel-estimation data loaders."""

from __future__ import annotations

import re


_METADATA_PATTERN = re.compile(
    r"^(?P<index>\d+)_SNR-(?P<snr>-?\d+)_DS-(?P<ds>\d+)_DOP-(?P<dop>\d+)_N-(?P<n>\d+)_(?P<profile>[A-Z\-]+)\.mat$"
)


def parse_metadata(file_name: str) -> dict[str, int | str]:
    """Parse benchmark metadata from the official AdaFortiTran file naming scheme."""

    match = _METADATA_PATTERN.match(file_name)
    if match is None:
        raise ValueError(f"unsupported channel-estimation filename: {file_name}")
    return {
        "index": int(match.group("index")),
        "snr": int(match.group("snr")),
        "delay_spread": int(match.group("ds")),
        "doppler": int(match.group("dop")),
        "pilot_spacing": int(match.group("n")),
        "profile": str(match.group("profile")),
    }
