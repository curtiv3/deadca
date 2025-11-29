"""
Stub script to apply an edit map inside DaVinci Resolve.

This script expects DaVinci Resolve's Python API to be available and will
add markers to the current timeline based on edit_map.json content.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


EDIT_MAP_PATH = Path(__file__).resolve().parent.parent / "examples" / "sample_edit_map.json"


def load_edit_map(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("edit_map.json must contain a list of edit actions.")
    return data


def get_timeline(resolve) -> Any:
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if project is None:
        raise RuntimeError("No active project in Resolve.")
    timeline = project.GetCurrentTimeline()
    if timeline is None:
        raise RuntimeError("No active timeline in Resolve.")
    return timeline


def apply_markers(timeline, edits: List[Dict[str, Any]]) -> None:
    for edit in edits:
        timecode = float(edit.get("time", 0))
        action = edit.get("action", "edit")
        label = edit.get("label", action)
        color = edit.get("color", "Red")
        timeline.AddMarker(timecode, color, label, "", 1)


def main() -> None:
    try:
        import DaVinciResolveScript  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "DaVinciResolveScript module not found. Please run this inside DaVinci Resolve's scripting environment."
        ) from exc

    resolve = DaVinciResolveScript.scriptapp("Resolve")
    timeline = get_timeline(resolve)
    edit_map = load_edit_map(EDIT_MAP_PATH)
    apply_markers(timeline, edit_map)
    print(f"Applied {len(edit_map)} markers from {EDIT_MAP_PATH}.")


if __name__ == "__main__":
    main()
