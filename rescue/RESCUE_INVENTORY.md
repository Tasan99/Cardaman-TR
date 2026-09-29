# Cardaman rescue snapshot — 2026-09-29

**Status:** safety snapshot of the files that remained after the OneDrive loss. Not a product development baseline.
Recovery status at snapshot time: `RECOVERY_INCOMPLETE`.

| | |
|---|---|
| Source (not modified, not moved) | `C:\Users\tasan\OneDrive\Masaüstü\Cardaman` |
| Destination | `C:\dev\Cardaman-rescue-20260929` |
| Method | `robocopy /E /COPY:DAT /DCOPY:T /R:1 /W:1 /XJ` (copy only; no `/MOVE`, no `/MIR`) |
| Copied at | 2026-09-29 23:29 (UTC+3) |
| Files source / destination | 376 / 376 |
| Bytes source / destination | 60,158,599 / 60,158,599 |
| SHA-256 checked | 376, missing 0, mismatch 0 (`hash-manifest.csv`) |
| `backend/src/regchain` files | 180 / 180 (89 `.py`, `workspace_ui.html`, ingestion CA certificate, `__pycache__`) |
| `backend/tests` files | 166 / 166 (82 `test_*.py` and `__pycache__`) |

`hash-manifest.csv` records the files as copied. The root `.gitignore` was extended after the copy (secrets, model
binaries, caches, logs), so its hash in the repository differs from the manifest.

## Present

- `backend/src/regchain/**`, `backend/tests/**`, `backend/pyproject.toml`
- Root: `.env` (not committed), `.env.example`, `.gitignore`, `compose.yaml`, `README.md`, `Cardaman.exe` (kept on disk,
  ignored by the existing `Cardaman*.exe` rule), `Cardamon.pdf`
- `evaluation/reports/baseline-v017-full/` (11 files)
- Empty directory skeletons of the lost folders (git does not record empty directories)

## Missing at snapshot time

- `backend/migrations/001_initial.sql` … `008_extraction_contract.sql`
- `docs/` (all)
- `scripts/` (all, including hardening/model-capability scripts, launchers and `launcher/CardamanLauncher.cs`)
- `evaluation/` except `reports/baseline-v017-full/`: datasets (`tr-aml-v1.json`), independent-v1, expert-gold-v1,
  fixtures, labels, reviewed_labels, capability, hardening-v020 (measurement snapshots, freezes, comparison, integrity,
  installed-model records), runs
- `samples/`, `data/`, `frontend/`, `contracts/`, `.github/`
- `.venv` (broken: no interpreter, no `pyvenv.cfg`)

## Local models (Ollama `/api/tags`, unchanged)

See `ollama-models.json`: `qwen3:4b` `359d7dd4…74fae7`, `qwen3:8b` `500a1f06…2b8b41`, `bge-m3:latest` `79076464…146bab`.

## Secret scan

No real credential found. Matches were template values (`local-*-change-me`, `disposable-test-only`) in `.env.example`
and `compose.yaml`, and intentional canaries in `test_v018_platform_regression.py` and `test_workspace.py`. `.env`
matches the example template and is excluded from git.

## Rule for recovered files

Recovered files are not merged silently into this snapshot. First a separate recovery inventory (counts, hashes, diff
against `hash-manifest.csv`), then a separate commit.
