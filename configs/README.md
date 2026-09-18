# Configuration

Only things that are genuinely configuration live here.

`data/*.yaml` is read by `carvision.config.load_data_config` and controls dataset
acquisition: which Hugging Face mirror, which columns, which split names, and the
validation fraction and seed. Change the mirror here and `carvision data download` uses
it. An unknown key is a hard error rather than a warning, so a typo cannot silently do
nothing.

**Backbones and heads are deliberately not configurable from YAML.** A backbone's
embedding width and preprocessing are properties of its pretrained weights, not
preferences, and the registry has to hold a factory function in Python regardless.
Putting the width in YAML as well would create a second source of truth that can only
drift. `carvision.models.backbones.REGISTRY` is the single source; select one with
`--backbone`.

This directory previously also held `backbone/`, `head/` and a Hydra `config.yaml`.
Nothing read any of them — including the mirror id that the download module's docstring
told you to edit. They were removed rather than left as decoration.
