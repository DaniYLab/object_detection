"""Serve the local, dependency-free room-gold annotation workbench."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.drawing_graph import AnnotationConflictError, RoomGoldStore  # noqa: E402


RECORD_ROUTE = re.compile(r"^/api/record/(\d+)$")
IMAGE_ROUTE = re.compile(r"^/api/image/([0-9A-Za-z_-]+)$")
DOOR_SEMANTIC_IDS = {1, 2, 3, 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("outputs/drawing_graph/room_gold_pack_v0.jsonl"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ArchCAD"))
    parser.add_argument(
        "--ui",
        type=Path,
        default=Path("tools/room_gold_annotator/index.html"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


class AnnotatorServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        store: RoomGoldStore,
        data_root: Path,
        ui_path: Path,
    ) -> None:
        super().__init__(address, AnnotatorHandler)
        self.store = store
        self.data_root = data_root.resolve()
        self.ui_path = ui_path.resolve()

    def door_instance_ids(self, image_id: str) -> list[str]:
        annotation_path = self.data_root / "json" / f"{image_id}.json"
        source = json.loads(annotation_path.read_text(encoding="utf-8"))
        return sorted(
            {
                str(entity["instance"])
                for entity in source.get("entities", [])
                if isinstance(entity, dict)
                and entity.get("semantic") in DOOR_SEMANTIC_IDS
                and entity.get("instance")
            }
        )


class AnnotatorHandler(BaseHTTPRequestHandler):
    server: AnnotatorServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _send_json(self, value: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self._send_json({"error": "file not found"}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_file(self.server.ui_path, "text/html; charset=utf-8")
            return
        if path == "/api/summary":
            self._send_json(self.server.store.summary())
            return
        record_match = RECORD_ROUTE.fullmatch(path)
        if record_match:
            index = int(record_match.group(1))
            try:
                record, version = self.server.store.get(index)
                doors = self.server.door_instance_ids(str(record["image_id"]))
            except IndexError:
                self._send_json({"error": "record index out of range"}, HTTPStatus.NOT_FOUND)
                return
            except (OSError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(
                {
                    "index": index,
                    "record_count": len(self.server.store),
                    "version": version,
                    "record": record,
                    "door_instance_ids": doors,
                }
            )
            return
        image_match = IMAGE_ROUTE.fullmatch(path)
        if image_match:
            image_id = image_match.group(1)
            if not self.server.store.contains_image(image_id):
                self._send_json({"error": "unknown image_id"}, HTTPStatus.NOT_FOUND)
                return
            image_path = (self.server.data_root / "png" / f"{image_id}.png").resolve()
            if not image_path.is_relative_to(self.server.data_root):
                self._send_json({"error": "invalid image path"}, HTTPStatus.BAD_REQUEST)
                return
            self._send_file(image_path, "image/png")
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        record_match = RECORD_ROUTE.fullmatch(path)
        if not record_match:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "invalid Content-Length"}, HTTPStatus.BAD_REQUEST)
            return
        if content_length <= 0 or content_length > 4 * 1024 * 1024:
            self._send_json({"error": "request body must be 1 byte to 4 MiB"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a mapping")
            version = payload.pop("version")
            record, new_version = self.server.store.update_editable_fields(
                int(record_match.group(1)), payload, expected_version=str(version)
            )
        except AnnotationConflictError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except IndexError:
            self._send_json({"error": "record index out of range"}, HTTPStatus.NOT_FOUND)
            return
        self._send_json(
            {
                "ok": True,
                "record": record,
                "version": new_version,
                "summary": self.server.store.summary(),
                "backup_path": (
                    str(self.server.store.backup_path)
                    if self.server.store.backup_path is not None
                    else None
                ),
            }
        )


def main() -> None:
    args = parse_args()
    store = RoomGoldStore(args.pack)
    server = AnnotatorServer(
        (args.host, args.port), store, args.data_root, args.ui
    )
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Room-gold annotator: {url}")
    print(f"Pack: {store.path}")
    print("Press Ctrl+C to stop.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotator.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
