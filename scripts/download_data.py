"""Download the camera video of the catalog artifact.
Usage: CATALOG_USER=... CATALOG_PASS=... python scripts/download_data.py [artifact_id]
"""
import json, os, subprocess, sys
from pathlib import Path
import requests

ART = sys.argv[1] if len(sys.argv) > 1 else "a35726a9-c637-4ab2-bd09-366ba9e1c7fe"
here = Path(__file__).parent
tok = json.loads(subprocess.check_output([sys.executable, str(here / "catalog_login.py")]))["tokens"]["access_token"]
d = requests.post("https://catalog.tbdrobotics.com/api/artifacts/describe", json={"artifact_identifier": ART},
                  headers={"Authorization": "Bearer " + tok}).json()
print("artifact:", d["metadata"]["name"])
out = here.parent / "data"; out.mkdir(exist_ok=True)
(out / "artifact_metadata.json").write_text(json.dumps({k: d[k] for k in ("metadata", "attributes")}, indent=2))
url = next(a["url"] for a in d["external_attachment"] if a["name"] == "Camera Video")
with requests.get(url, stream=True) as r, open(out / "camera_video.mp4", "wb") as f:
    r.raise_for_status()
    for chunk in r.iter_content(1 << 20):
        f.write(chunk)
print("saved", out / "camera_video.mp4")
