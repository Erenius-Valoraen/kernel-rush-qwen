from pathlib import Path

from client import Dryft
from package import package

LATENCY_GATE = 1.10

ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"


def report(detail: dict) -> bool:
    state = detail.get("state")
    result = detail.get("result") or {}
    shapes = result.get("shapes") or []

    print(f"run {detail.get('id')}: {state}")
    if result.get("score") is not None:
        print(f"score {result['score']:.1f}  (100 is native)")

    for shape in shapes:
        metrics = shape.get("modelMetrics") or {}
        columns = [f"{shape['id']:<10}", f"{shape.get('caseStatus', '?'):<13}"]
        if shape.get("metricMs") and metrics.get("referenceMs"):
            speedup = metrics["referenceMs"] / shape["metricMs"]
            columns.append(f"{shape['metricMs']:8.1f} ms  {speedup:5.2f}x native")
        for name, mine, native in (
            ("ttft", metrics.get("ttftMs"), metrics.get("referenceTtftMs")),
            ("tpot", metrics.get("tpotMs"), metrics.get("referenceTpotMs")),
        ):
            if mine and native:
                ratio = mine / native
                flag = "  OVER GATE" if ratio > LATENCY_GATE else ""
                columns.append(f"{name} {ratio:4.2f}x{flag}")
        if shape.get("tokensPerSecond"):
            columns.append(f"{shape['tokensPerSecond']:7.1f} tok/s")
        print("  " + "  ".join(columns))
        if shape.get("caseMessage"):
            print(f"    {shape['caseMessage']}")

    for label, value in (
        ("failure", result.get("failureMessage") or result.get("failureCode")),
        ("error", detail.get("errorMessage") or detail.get("errorCode")),
        ("not ranked", result.get("rankingReason")),
    ):
        if value:
            print(f"  {label}: {value}")

    return state == "succeeded" and all(
        shape.get("caseStatus") != "failed" for shape in shapes
    )


def attempt(client: Dryft, engine_dir: Path, mode: str, timeout: float) -> bool:
    archive = package(engine_dir)
    print(f"packaged {engine_dir} -> {len(archive)} bytes")

    submission_id = client.submit(archive)
    started = client.start_run(submission_id, mode=mode)
    run_id = started["id"]
    print(f"submission {submission_id}, {mode} run {run_id}; waiting")

    return report(client.wait(run_id, timeout=timeout))


def plan_next_edit(history: list[dict]) -> str:
    raise NotImplementedError("this is the part you write")
