from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
from serial.tools import list_ports

SO100_USB_VID = 0x1A86  # 6790
KNOWN_SO100_PIDS = {
    0x7523,
    0x55E3,
    0x5523,
    0x5525,
    0x55D3,
}  # your ints mapped to hex if desired

# Strong signals (tune to what you actually see in list_ports output)
GOOD_PRODUCT_TOKENS = (
    "ch340",
    "ch341",
    "wch",
    "usb serial",
    "usb-serial",
    "single serial",
    "cdc acm",
)
BAD_PRODUCT_TOKENS = ("bluetooth", "gps", "modem", "wwan", "diag")


@dataclass(frozen=True)
class So100PortSelection:
    preferred_serial_by_id: Optional[str] = None  # exact serial number if known
    preferred_by_id_contains: Optional[str] = (
        None  # substring in /dev/serial/by-id link name
    )
    preferred_usb_location: Optional[str] = (
        None  # p.location string (USB topology) if known
    )


def _find_stable_symlink(dev: str) -> Optional[str]:
    dev_real = os.path.realpath(dev)
    for base in ("/dev/serial/by-id", "/dev/serial/by-path"):
        if not os.path.isdir(base):
            continue
        for link in glob.glob(f"{base}/*"):
            try:
                if os.path.realpath(link) == dev_real:
                    return link
            except OSError:
                pass
    return None


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def find_so100_port(
    prefs: So100PortSelection = So100PortSelection(),
    allow_fallback_vid_only: bool = False,
) -> str:
    ports = list(list_ports.comports())

    candidates = []
    for p in ports:
        if p.vid is None or p.pid is None:
            continue

        vid = int(p.vid)
        pid = int(p.pid)

        # Tight base filter: require VID match AND (PID match unless you allow fallback)
        if vid != SO100_USB_VID:
            continue
        if (pid not in KNOWN_SO100_PIDS) and (not allow_fallback_vid_only):
            continue

        dev = p.device
        stable = _find_stable_symlink(dev)

        product = _norm(getattr(p, "product", None))
        manufacturer = _norm(getattr(p, "manufacturer", None))
        hwid = _norm(getattr(p, "hwid", None))
        serial = _norm(getattr(p, "serial_number", None))
        location = _norm(getattr(p, "location", None))

        # Hard negative: looks like modem-ish device
        if any(tok in product for tok in BAD_PRODUCT_TOKENS) or any(
            tok in hwid for tok in BAD_PRODUCT_TOKENS
        ):
            continue

        # Scoring: highest wins
        score = 0

        # Exact pinning (if you can provide it)
        if prefs.preferred_serial_by_id and serial == _norm(
            prefs.preferred_serial_by_id
        ):
            score += 10_000
        if prefs.preferred_usb_location and location == _norm(
            prefs.preferred_usb_location
        ):
            score += 5_000
        if (
            prefs.preferred_by_id_contains
            and stable
            and _norm(prefs.preferred_by_id_contains) in _norm(stable)
        ):
            score += 5_000

        # Prefer stable symlink paths (by-id > by-path > raw)
        if stable:
            score += 1_000
            if "/by-id/" in stable:
                score += 200

        # Prefer ports that look like USB-serial adapters we expect
        if any(tok in product for tok in GOOD_PRODUCT_TOKENS):
            score += 200
        if any(tok in manufacturer for tok in ("wch", "qinheng", "usb")):
            score += 50
        if serial:
            score += 100

        # As final tiebreakers, prefer ttyUSB over ttyACM only if you know that’s your controller
        if dev.startswith("/dev/ttyUSB"):
            score += 5
        elif dev.startswith("/dev/ttyACM"):
            score += 0

        candidates.append((score, stable or dev, p))

    if not candidates:
        raise RuntimeError("No SO-100 candidate serial port found (strict match).")

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path, best_port = candidates[0]

    # If multiple candidates are close, fail fast so you don't randomly pick the wrong one
    if len(candidates) > 1 and (best_score - candidates[1][0]) < 100:
        details = [
            f"{s}: {path} (dev={p.device}, vid={p.vid}, pid={p.pid}, serial={getattr(p, 'serial_number', None)}, product={getattr(p, 'product', None)}, location={getattr(p, 'location', None)})"
            for s, path, p in candidates[:5]
        ]
        raise RuntimeError(
            "Ambiguous SO-100 serial port selection; pin it via serial number or USB location.\n"
            + "\n".join(details)
        )

    return best_path


def get_camera_info(index_or_path: Union[int, str, Path]) -> Tuple[int, int, int]:
    cap = cv2.VideoCapture(str(index_or_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at {index_or_path}")

    # Read a frame to make sure the stream is active
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Failed to read from camera at {index_or_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cap.release()
    return width, height, fps
