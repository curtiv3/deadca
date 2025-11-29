# deadcase_tools

Toolkit for building the DeadCaseAI horror video workflow. It currently supports shot detection with simple metadata, generating edit maps via the OpenAI API, and a stub for applying edits inside DaVinci Resolve.

## Structure
- `deadcase_tools/analyze_video.py` – Analyze a video to detect shots and compute brightness/motion.
- `deadcase_tools/deadcase_director.py` – Send shot data to OpenAI to generate an edit map.
- `deadcase_tools/examples/` – Sample `shots.json` and `edit_map.json` outputs.
- `deadcase_tools/resolve/apply_edits_in_resolve.py` – Stub script for adding markers in DaVinci Resolve.

## Setup
Install dependencies (OpenCV and OpenAI):

```bash
pip install opencv-python openai numpy
```

Ensure your OpenAI API key is available as `OPENAI_API_KEY` in the environment.

## Usage
Run shot analysis:

```bash
python deadcase_tools/analyze_video.py --input input.mp4 --output shots.json
```

Generate an edit map:

```bash
python deadcase_tools/deadcase_director.py --shots shots.json --output edit_map.json
```

(Stub) Apply edits in DaVinci Resolve (from within Resolve's scripting console):

```bash
python deadcase_tools/resolve/apply_edits_in_resolve.py
```

## Examples
See `deadcase_tools/examples/sample_shots.json` and `deadcase_tools/examples/sample_edit_map.json` for example outputs.
