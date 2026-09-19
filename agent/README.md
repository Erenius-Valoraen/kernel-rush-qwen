# The agent side

Nothing in this folder is submitted. It runs on your machine, drives the API,
and is where your autoresearch loop lives.

For interactive use, run `../install-dryft.sh` (or `../install-dryft.ps1` on
Windows) to install `../bin/dryft`. The files here are small, dependency-free
Python building blocks for an automated research loop.

| File | What it does |
| --- | --- |
| `package.py` | Builds the archive from `engine/`, and refuses what the platform would refuse. |
| `client.py` | The submission API over the standard library. No dependencies. |
| `loop.py` | One turn of a research loop: package, submit, run, print what it measured. |

Submit and wait with the installed CLI. It already knows the event server, so
most people only need `DRYFT_TOKEN`:

```sh
export DRYFT_TOKEN=dryft_pat_...                   # API tokens

../bin/dryft submit engine
../bin/dryft run <submission-id> --mode public --wait 3000
../bin/dryft run <submission-id> --mode official --wait 3000
```

Set `DRYFT_API` only if `client.py` is talking to a local, staging, or
self-hosted server (no `/api` suffix). The CLI does not need it for the
public event.

`report` in `loop.py` prints a row per workload with the measured time, the
speedup over native, and the time-to-first-token and time-per-output-token
ratios. Those two ratios are gates on an official run — above 1.10 and the
workload fails — but a public run only reports them, so read them before you
promote a change.

`plan_next_edit` in `loop.py` is the part you write: given what previous
attempts measured, decide what to change about `engine/engine.py`. Keep a
record of every attempt. Only the three hidden workloads are scored, and the
public three will not always explain why a score moved.
