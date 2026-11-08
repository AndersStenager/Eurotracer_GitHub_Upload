"""The complete public workflow: prepare -> place -> operate -> flow -> evaluate."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import operations, placement
from . import storage as st
from .config import Scenario, Weather, positive
from .data import prepare

if TYPE_CHECKING:
    from .results import Run


def run(
    project: str | Path,
    scenario: Scenario | str | None = None,
    *,
    inputs: str = "input1",
    years: tuple[int, int] | None = None,
    start: str | None = None,
    stop: str | None = None,
    name: str = "baseline",
    weather: Weather | None = None,
    archive: str | Path | None = None,
    memory_gib: float = 24,
    download: bool = True,
    progress: bool = True,
) -> Run:
    """Design and operate a scenario, reusing completed physical calculations.

    Year bounds are inclusive. Timestamp bounds use UTC and an exclusive stop.
    memory_gib bounds numerical scratch estimates, not total machine memory.
    """
    from .results import Run

    project = Path(project).resolve()
    name = st.safe_name(name)
    scenario = scenario or Scenario()
    if isinstance(scenario, str):
        scenario = Scenario(scenario)
    positive(memory_gib, "memory_gib")
    if memory_gib < 0.25:
        raise ValueError("At least 0.25 GiB working memory is required")
    started = time.perf_counter()
    data = prepare(
        project,
        inputs=inputs,
        years=years,
        start=start,
        stop=stop,
        weather=weather or Weather(),
        archive=archive,
        offshore=scenario.offshore is not None,
        download=download,
        progress=progress,
    )
    layout = placement.solve(data, scenario, progress)
    operation = operations.operate(data, scenario, layout, memory_gib=memory_gib, progress=progress)
    flow = operations.power_flow(
        data, scenario, layout, operation, memory_gib=memory_gib, progress=progress
    )
    definition = {
        "scenario": st.plain(scenario),
        "prepared": str(data.path),
        "layout": str(layout),
        "operations": str(operation),
        "flow": str(flow),
        "format": st.FORMAT,
    }
    key = st.identity(definition)
    path = project / "runs" / name / key
    with st.lock(path):
        if not (path / "run.json").exists():
            st.write_json(
                path / "run.json",
                {
                    **definition,
                    "id": key,
                    "name": name,
                    "project": project,
                    "memory_gib": memory_gib,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
    st.write_json(project / "runs" / name / "latest.json", {"id": key})
    result = Run(path)
    if progress:
        print(f"Run {name}/{key}: ready in {time.perf_counter() - started:.1f}s")
        print(result.summary.to_string(index=False))
    return result


def open_run(project: str | Path, name: str | None = None, *, revision: str | None = None) -> Run:
    """Open a revision directory, or a project's latest named run, without scans."""
    from .results import Run

    path = Path(project).resolve()
    if name is not None:
        path = path / "runs" / st.safe_name(name)
    if (path / "run.json").exists():
        return Run(path)
    key = revision or st.read_json(path / "latest.json")["id"]
    return Run(path / st.safe_name(key))
