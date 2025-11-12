from __future__ import annotations

from pathlib import Path

from tabletop.logging.pupylabs_cloud import PupylabsCloudLogger
from tabletop.utils.http_client import ApiDnsError


class _StubClient:
    def __init__(self) -> None:
        self.base_url = "https://cloud.test"
        self.fail = True
        self.calls: list[str] = []

    def close(self) -> None:  # pragma: no cover - no-op
        pass

    def health_check(self, paths, **_) -> str:
        if self.fail:
            raise ApiDnsError("dns fail")
        return paths[0]

    def post(self, path, **_) -> object:
        self.calls.append(path)
        if self.fail:
            raise ApiDnsError("dns fail")

        class _Response:
            status_code = 200

        return _Response()


def test_offline_queue_flushes(tmp_path: Path) -> None:
    stub = _StubClient()
    queue_file = tmp_path / "queue.ndjson"
    logger = PupylabsCloudLogger(stub, api_key="test", queue_path=queue_file)

    event = {"event_id": "1"}
    logger.send(event)

    assert queue_file.exists()
    assert queue_file.read_text(encoding="utf-8").strip()
    assert not stub.calls

    stub.fail = False
    logger.flush()

    assert stub.calls == ["/v1/events/ingest"]
    assert queue_file.exists()
    assert queue_file.read_text(encoding="utf-8") == ""
