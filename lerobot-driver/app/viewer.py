import hashlib
import os
import uuid

import rerun as rr


def _deterministic_uuid_v4_from_string(val: str) -> uuid.UUID:
    h = hashlib.sha256(val.encode()).digest()
    b = bytearray(h[:16])
    b[6] = (b[6] & 0x0F) | 0x40  # Version 4
    b[8] = (b[8] & 0x3F) | 0x80  # Variant RFC 4122
    return uuid.UUID(bytes=bytes(b))


def init_rerun_viewer_server(
    *,
    system_id: str | None = None,
    memory_limit: str | None = None,
    newest_first: bool = True,
) -> rr.RecordingStream:
    """
    Starts a Rerun recording stream + gRPC server, and (optionally) a web viewer.

    Env defaults:
      - SYSTEM_ID (fallback: "lerobot-drive")
      - RERUN_GRPC_PORT (fallback: 9876)
      - RERUN_HTTP_PORT (fallback: 9877)
      - RERUN_MEMORY_LIMIT (fallback: "25%")
      - RERUN_NEWEST_FIRST (fallback: "true")

    Returns:
      rr.RecordingStream you can rr.log(...) into.
    """
    if system_id is None:
        system_id = os.getenv("SYSTEM_ID", "lerobot-drive")
    if memory_limit is None:
        memory_limit = os.getenv("RERUN_MEMORY_LIMIT", "25%")

    if newest_first is True:
        newest_first = os.getenv("RERUN_NEWEST_FIRST", "true").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    recording = rr.RecordingStream(
        application_id=system_id,
        recording_id=_deterministic_uuid_v4_from_string(system_id),
    )

    rr.spawn(
        server_memory_limit=memory_limit,
        recording=recording,
    )

    return recording
