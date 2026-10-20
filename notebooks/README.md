# Interactive thesis notebook

`thesis_interactive.py` is a marimo notebook with ten interactive diagrams of the
thesis, built only from the precomputed JSON files in `data/`. It loads no model
and needs no GPU.

Run it locally:

```
pip install marimo altair pandas numpy chess
marimo run notebooks/thesis_interactive.py
```

When the notebook runs without the `data/` folder beside it (for example
mirrored into molab from its GitHub URL), it reads the same files from this
repository on GitHub instead.

Held-out test games carry only non-identifying PGN headers: human players are
shown as "Anonymous".
