import hashlib
import logging
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
    if system_id is None:
        system_id = os.getenv("SYSTEM_ID", "lerobot-drive")
    if memory_limit is None:
        memory_limit = os.getenv("RERUN_MEMORY_LIMIT", "25%")

    newest_first = os.getenv("RERUN_NEWEST_FIRST", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    grpc_port = int(os.getenv("RERUN_GRPC_PORT", "9876"))
    http_port = int(os.getenv("RERUN_HTTP_PORT", "9877"))

    recording = rr.RecordingStream(
        application_id=system_id,
        recording_id=_deterministic_uuid_v4_from_string(system_id),
    )
    rr.set_global_data_recording(recording)

    # 1) Start gRPC server that buffers data for late-connecting viewers
    server_uri = rr.serve_grpc(
        grpc_port=grpc_port,
        recording=recording,
        server_memory_limit=memory_limit,
    )
    # server_uri is usually like: rerun+http://0.0.0.0:9876/proxy

    # 2) Serve the web viewer over HTTP and make it connect to the gRPC server
    # NOTE: this API exists in rerun>=0.24 (serve_web() is deprecated).
    rr.serve_web_viewer(
        web_port=http_port,
        open_browser=False,
        connect_to=server_uri,
    )

    attach_rerun_logging()

    return recording


class RerunLoggingHandler(logging.Handler):
    LEVEL_MAP = {
        logging.DEBUG: rr.TextLogLevel.DEBUG,
        logging.INFO: rr.TextLogLevel.INFO,
        logging.WARNING: rr.TextLogLevel.WARN,
        logging.ERROR: rr.TextLogLevel.ERROR,
        logging.CRITICAL: rr.TextLogLevel.ERROR,
    }

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            level = self.LEVEL_MAP.get(record.levelno, rr.TextLogLevel.INFO)

            # Use logger name as entity path so logs are grouped
            entity_path = f"logs/{record.name}"

            rr.log(
                entity_path,
                rr.TextLog(msg, level=level),
            )
        except Exception:
            pass  # never let logging crash the app


def attach_rerun_logging(level=logging.INFO):
    handler = RerunLoggingHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
