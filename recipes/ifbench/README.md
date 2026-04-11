# IFBench

`recipes/ifbench/_vendor` contains vendored checker code and should stay in git.
Only `recipes/ifbench/_vendor/.nltk_data/` is treated as local runtime data and is git-ignored.

## One-time Setup

The vendored IFBench checkers look for NLTK resources under `recipes/ifbench/_vendor/.nltk_data`.
Download them once before running the recipe:

```bash
python - <<'PY'
from pathlib import Path

import nltk

target = Path("recipes/ifbench/_vendor/.nltk_data")
target.mkdir(parents=True, exist_ok=True)

for resource in (
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger_eng",
):
    nltk.download(resource, download_dir=str(target), quiet=False)
PY
```

If you already have these NLTK resources elsewhere, you can also copy them into
`recipes/ifbench/_vendor/.nltk_data/`.
