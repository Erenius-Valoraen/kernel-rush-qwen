# Decode Qwen3 4B faster

Decode `Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, BF16, on one H100

`engine/engine.py` is the baseline: Transformers, BF16, greedy decode with a KV
cache.

Read the [Docs](https://htn.dryft.ai/docs) before you optimize
(workloads, timing, output rule, scoring). `QWEN_ENGINE_CONTRACT.md` is the same
guide for offline use. Then read `OPTIMIZATION_GUIDE.md` for the pinned model
architecture, tensor shapes, execution graph, and staged examples of replacing
Transformers operations.

## Get started

1. Fork [Dryft-Kernels/starter](https://github.com/Dryft-Kernels/starter)
2. Clone the new repository.
3. [Sign in to Dryft](https://htn.dryft.ai/) with GitHub.
   Create a team, or join one with a six-character invite code
4. [Submissions](https://htn.dryft.ai/bench), select
   **Connect a repository**, grant the GitHub App access, and set
   **Engine folder** to `engine`.
5. Push to the default branch, or use **Run now** in
   [Repositories](https://htn.dryft.ai/repos). Check the public
   sample results before changing the engine.

Public runs are feedback. Request an official evaluation to appear on the
leaderboard.

No local GPU or Python setup is required. Edit `engine/engine.py` and push
again. For CLI submissions, see [Submitting](#submitting).

## Layout

| Path | Submitted | What it is |
| --- | --- | --- |
| `engine/engine.py` | yes | Your engine. Native Qwen until you replace it. |
| `engine/kernels/` | yes | Worked Triton example. Delete it or build on it. |
| `agent/` | no | Autoresearch loop. Runs on your machine. |
| `bin/` | no | Installed Dryft CLI. |
| `requirements.txt` | no | Container versions, for a local GPU. |
| `AGENTS.md` | no | Contract as rules, for a coding agent. |
| `OPTIMIZATION_GUIDE.md` | no | Architecture, tensor shapes, and replacement map. |

Only `engine/` is submitted. Keep the agent, notes, and credentials outside it.

## Engine interface

The archive root must hold `engine.py`, which exports `class Engine` with
exactly two methods. Dryft names repository submissions from the repository and
your team’s stable slot, such as `fast-qwen #7`.

```python
class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yield a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence in input_ids has the
        same length. Do not stop at end-of-sequence tokens.
        """
```

Put any Python or Triton files the engine imports beside `engine.py`. The run
container has no network, so the engine cannot install or download anything.

## Submitting

From this folder, install the CLI. It downloads the public binary, checks it,
and puts it in `bin/`:

```sh
./install-dryft.sh

# Windows PowerShell:
# .\install-dryft.ps1
```

Create a token under [API tokens](https://htn.dryft.ai/tokens),
then:

```sh
export DRYFT_TOKEN='dryft_pat_...'

./bin/dryft doctor
./bin/dryft validate engine
./bin/dryft submit engine
./bin/dryft run <submission-id> --mode public --wait 3000
```

`submit` prints the submission ID that `run` needs. The CLI already knows the
event server, so most people only need `DRYFT_TOKEN`. Set `DRYFT_API` only for
local development, staging, or a self-hosted server.

```sh
./bin/dryft submissions                  # latest 25 submissions
./bin/dryft runs                         # latest 25 runs
./bin/dryft logs <run-id> --follow
./bin/dryft result <run-id> --wait --timeout 3000
./bin/dryft cancel <run-id> --reason 'superseded'
```

The CLI accepts the engine folder, `engine.py`, or an existing `.tar.gz`. To
package by hand, run this from the starter folder:

```sh
cd engine && tar -czf ../submission.tar.gz engine.py kernels
```

Name the files explicitly. `tar -C engine .` writes paths such as `./engine.py`,
which the platform rejects.

Limits: 2 MiB compressed, 16 MiB expanded, 200 files. Allowed extensions:
`.py .pyi .yaml .yml .json .toml .txt .md .cfg .ini`. No weights, credentials,
compiled binaries, Docker images, or agent code.
