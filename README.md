# asf_home_electrification

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for virtual environment management. If you are new to `uv`, you can find the [quickstart guide here](https://docs.astral.sh/uv/getting-started/).

We also utilise `direnv` via the `.envrc` file to automatically:

- Import your environment variables from `.env`
- Activate your virtual environment (_only if you comment out the relevant lines in `.envrc`_)

After installing `direnv` and `uv` on your system (we recommend doing this via [`brew`](https://brew.sh/) on macOS), you **must** run the following commands in your terminal to set up the project:

```bash
direnv allow
uv sync
uv run pre-commit install --install-hooks
```

### UV private index failsafe

`unset UV_INDEX` is enabled by default in `.envrc` to prevent private index settings from leaking into `uv.lock`. If you need to use a private `UV_INDEX` in this project, comment out the `unset UV_INDEX` line.

If a lockfile has already been generated with a private index, use this rollback/failsafe sequence:

```bash
unset UV_INDEX
uv sync --no-cache --upgrade
# or
uv lock --no-cache --upgrade
```

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

### Versioning

We use [Semantic Versioning](https://semver.org/) for versioning. This project's versioning is inferred dynamically at build via `hatchling`, based on the value in the base `__init__.py` file.

To update and release the package:

1. Update the version in `asf_home_electrification/__init__.py` (and any other relevant places)
2. Open a release PR with the version changes
3. Once approved and merged, create a new release and tag on GitHub with the matching version number
4. User's can then install the package via `uv add <repository URL> --tag <version>`

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
