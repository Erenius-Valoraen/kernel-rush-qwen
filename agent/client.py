import contextlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

#: Run states that will never change again.
TERMINAL = frozenset({"succeeded", "failed", "timed_out", "canceled", "infra_error"})


class ApiError(RuntimeError):
    """The API refused a request, or reported one of its own failures."""

    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(f"{code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


class Dryft:
    def __init__(self, base_url: str | None = None, token: str | None = None):
        base = base_url or os.environ.get("DRYFT_API") or ""
        if not base:
            raise RuntimeError("set DRYFT_API to your deployment's origin, without /api")
        if urllib.parse.urlsplit(base).scheme not in ("http", "https"):
            raise RuntimeError(f"DRYFT_API must be an http(s) origin, not {base!r}")
        self.base_url = base.rstrip("/")
        self.token = token or os.environ.get("DRYFT_TOKEN") or ""
        if not self.token:
            raise RuntimeError("set DRYFT_TOKEN to a team API token from Team → API tokens")

    def _send(self, method: str, path: str, *, body: bytes | None = None,
              content_type: str | None = None, extra_headers: dict | None = None):
        request = urllib.request.Request(  # noqa: S310 - scheme checked in __init__
            f"{self.base_url}{path}", data=body, method=method,
        )
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("Accept", "application/json")
        if content_type:
            request.add_header("Content-Type", content_type)
        for name, value in (extra_headers or {}).items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            payload = {}
            with contextlib.suppress(ValueError):
                payload = json.loads(error.read() or b"{}")
            problem = payload.get("error") or {}
            raise ApiError(
                error.code,
                problem.get("code", "http_error"),
                problem.get("message", error.reason or "request failed"),
                problem.get("details"),
            ) from None

    def benchmark(self) -> dict:
        """The public benchmark definition: workloads, budgets, judging rule.

        There is one benchmark, so the listing has one item and nothing has to
        name it.
        """
        items = self._send("GET", "/api/v1/challenges").get("items") or []
        if not items:
            raise RuntimeError("the platform has no published benchmark")
        return items[0]

    def submit(self, archive: bytes) -> str:
        """Upload an archive. Returns its submission id."""
        boundary = f"----dryft{uuid.uuid4().hex}"
        body = b"".join([
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="archive"; '
            b'filename="submission.tar.gz"\r\n',
            b"Content-Type: application/gzip\r\n\r\n",
            archive,
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        result = self._send(
            "POST", "/api/v1/submissions", body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        return result["submission"]["id"]

    def start_run(self, submission_id: str, mode: str = "public",
                  idempotency_key: str | None = None) -> dict:
        """Start a run. Reuse the same key to replay rather than duplicate one."""
        if mode not in ("public", "official"):
            raise ValueError("mode is 'public' or 'official'")
        result = self._send(
            "POST", f"/api/v1/submissions/{submission_id}/runs",
            body=json.dumps({"mode": mode}).encode(),
            content_type="application/json",
            extra_headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )

        return result["run"]

    def run(self, run_id: str) -> dict:
        return self._send("GET", f"/api/v1/runs/{run_id}")["run"]

    def logs(self, run_id: str, after: int = -1, limit: int = 200) -> dict:
        query = urllib.parse.urlencode({"after": after, "limit": limit})
        return self._send("GET", f"/api/v1/runs/{run_id}/logs?{query}")

    def wait(self, run_id: str, timeout: float = 3000, interval: float = 10) -> dict:
        """Poll until the run reaches a terminal state, or raise on timeout.

        A timeout here does not cancel anything. Keep the id and look again;
        never start a second run because the first one was slow to answer.
        """
        deadline = time.monotonic() + timeout
        while True:
            current = self.run(run_id)
            if current.get("state") in TERMINAL:
                return current
            if time.monotonic() >= deadline:
                raise TimeoutError(f"run {run_id} still {current.get('state')} after {timeout}s")
            time.sleep(interval)
