"""
Generate an edit map for DeadCaseAI by sending shot metadata to the OpenAI API.

Usage:
    python deadcase_director.py --shots shots.json --output edit_map.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

MODEL_NAME = "gpt-4.1"  # Placeholder; adjust as desired.


SYSTEM_MESSAGE = (
    "You are DeadCaseAI, a horror video director. "
    "Given a list of shots with metadata, return a JSON array of edit actions (cut, glitch, flash, zoom, marker) "
    "that enhance a dark, eerie mood without overwhelming the viewer."
)


RULES_TEMPLATE: Dict[str, Any] = {
    "max_cuts_per_minute": 12,
    "glitch_on_high_motion_dark_shots": True,
    "flash_on_brightness_spike": True,
    "default_flash_color": "white",
    "notes": "Keep pacing tense but readable; avoid jump cuts every frame."
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a DeadCaseAI edit map using the OpenAI API")
    parser.add_argument("--shots", required=True, help="Path to the shots JSON produced by analyze_video.py")
    parser.add_argument("--output", required=True, help="Path to the output edit_map.json")
    return parser.parse_args()


def load_shots(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Shots file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Shots JSON should be a list of shot objects.")
    return data


def call_openai(shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "shots": shots,
                        "rules": RULES_TEMPLATE,
                        "transcript": None,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    message_content = response.choices[0].message.content
    if not message_content:
        raise ValueError("Empty response from OpenAI API.")

    parsed: Dict[str, Any] = json.loads(message_content)
    edits = parsed.get("edits") if isinstance(parsed, dict) else None
    if edits is None:
        raise ValueError("OpenAI response missing 'edits' field.")
    if not isinstance(edits, list):
        raise ValueError("OpenAI response 'edits' should be a list.")
    return edits


def save_edit_map(edits: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(edits, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    shots_path = Path(args.shots)
    output_path = Path(args.output)

    try:
        shots = load_shots(shots_path)
        edits = call_openai(shots)
        save_edit_map(edits, output_path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to generate edit map: {exc}") from exc


if __name__ == "__main__":
    main()
