from pathlib import Path
from typing import Dict, Union, Tuple
import cv2

from serial.tools import list_ports

SO100_USB_VID = 6790
KNOWN_SO100_PIDS = {
    29987,
    21987,
    21795,
    21797,
    21971
}


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


def find_so100_port(index: int = 0) -> str:
    """Scan for SO-100 USB serial devices and return the port at the given index."""
    all_ports = list(list_ports.comports())
    matching_ports = [
        p.device
        for p in all_ports
        if (p.vid is not None and p.pid is not None) and
           (int(p.vid) == SO100_USB_VID or int(p.pid) in KNOWN_SO100_PIDS)
    ]
    if not matching_ports:
        raise RuntimeError("No SO-100 USB device found.")
    if index >= len(matching_ports):
        raise RuntimeError(f"Requested index {index}, but only {len(matching_ports)} SO-100 device(s) found.")
    return matching_ports[index]
