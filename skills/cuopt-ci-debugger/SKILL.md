---
name: cuopt-ci-debugger
version: "26.06.00"
description: Debug GitHub CI failures from a PR number, branch name, run id, job name, or failing matrix. Use the GitHub CLI to inspect checks, failed logs, PR diffs, and local code, then produce a concise root-cause analysis and rerun plan.
---

# cuOpt CI Debugger

Use this skill when the user provides a PR number, branch, run id, job name, failing matrix, or CI log excerpt and asks for debugging, triage, or automated analysis.

## Principles

- Prefer `gh` CLI over browser scraping.
- Do not assume "latest"; query the PR/run and use concrete run ids, job names, and timestamps.
- Start from the failing job logs, then correlate with the PR diff. Avoid guessing from the job name alone.
- Separate infrastructure failures from product failures before proposing code changes.
- Do not install or upgrade packages. If `gh` is missing or unauthenticated, report the exact blocker and ask the user to provide logs or authenticate locally.
- Do not push, rerun CI, close checks, or mutate remote state unless the user explicitly asks.

## Inputs

Accept any of:

- PR number, for example `#1037`
- branch name, for example `pull-request-branch` or the current checked-out branch
- run id or URL
- job name or matrix, for example `wheel-tests-cuopt-server / 13.1.1, 3.14, amd64, ubuntu24.04`
- failing test name
- CI log excerpt

If only a branch name is provided, find the matching PR first. If only a PR number is provided, discover the branch, checks, and most recent failing run with `gh`.

## Workflow

Run from the repo root.

### 1. Set Repo Context

```bash
git remote -v
gh repo set-default NVIDIA/cuopt
```

Use `gh repo set-default NVIDIA/cuopt` when `gh` reports no default repository or multiple remotes make the target ambiguous.

### 2. Resolve PR

If a PR number is known:

```bash
gh pr view <PR> -R NVIDIA/cuopt --json number,title,url,state,isDraft,headRefName,baseRefName,mergeStateStatus,statusCheckRollup,commits,files
```

If owner and branch are known, prefer the direct fork selector:

```bash
gh pr view <owner>:<branch> -R NVIDIA/cuopt --json number,title,url,state,isDraft,headRefName,baseRefName,mergeStateStatus,statusCheckRollup,commits,files
```

If only a branch is known, infer the owner locally, then use `owner:branch`:

```bash
branch=<branch>
git branch -r --list "*/$branch"
remote=$(git config branch."$branch".remote)
git remote get-url "$remote"
```

If branch lookup is still ambiguous, fall back to listing PRs:

```bash
gh pr list --head <branch> --state all -R NVIDIA/cuopt --json number,title,url,state,isDraft,headRefName,baseRefName,updatedAt
```

For the current branch:

```bash
branch=$(git branch --show-current)
```

### 3. Inspect CI

```bash
gh pr checks <PR> -R NVIDIA/cuopt
gh run list --branch <headRefName> --limit 20 -R NVIDIA/cuopt --json databaseId,name,status,conclusion,createdAt,updatedAt,headBranch,event
```

For a suspicious run:

```bash
gh run view <run_id> -R NVIDIA/cuopt --json databaseId,name,status,conclusion,createdAt,updatedAt,jobs
gh run view <run_id> -R NVIDIA/cuopt --log-failed
```

If `--log-failed` is too broad or truncated:

```bash
gh run view <run_id> -R NVIDIA/cuopt --json jobs
gh run view <run_id> -R NVIDIA/cuopt --job <job_id> --log
```

Use `gh api` only when higher-level commands do not expose enough detail:

```bash
gh api repos/NVIDIA/cuopt/actions/runs/<run_id>/jobs
```

### 4. Correlate Locally

```bash
git status --short
git diff --stat <base>...HEAD
git diff <base>...HEAD -- <suspect_paths>
rg "<error text|test name|symbol>"
```

Use local history when helpful:

```bash
git blame -L <start>,<end> -- <file>
git log --oneline -- <file>
```

## Log Triage Checklist

Extract these facts before editing code:

- workflow name, run id, job id, and matrix
- failing command
- failing test names
- first meaningful exception or assertion
- shortest stack frame pointing into project code
- whether setup/build/import failed before tests ran
- whether other jobs with nearby coverage passed

Classify the failure:

- **C++ only:** likely native logic, CUDA, gtest, or dataset/build issue.
- **Python only:** likely wrapper, validation, API behavior, Python tests, or package import issue.
- **Wheel only:** likely packaging, dependency pin, generated artifacts, ABI, install layout, or wheel test environment issue.
- **Server only:** likely REST schema, server validation, serialization, fixture, startup/healthcheck, or async polling issue.
- **Docs/style only:** likely formatting, docs build, lint, or metadata.
- **Many unrelated jobs fail at setup:** likely infrastructure, dependency, image, or runner issue.

## Repository Heuristics

- If C++ tests pass but Python, wheel, or server tests fail, inspect wrappers, generated bindings, package layout, server schemas, fixtures, and validation differences before changing native internals.
- Wheel failures often expose differences between built artifacts and local editable imports. Confirm whether the failing stack uses installed wheel paths or the workspace.
- Server failures often involve request schema validation, serialization, startup/healthcheck, async polling, or installed package layout.
- If a new native validation fails only in Python/server CI, compare the native bounds and semantics to existing Python/server validation before deciding whether fixtures or native checks are wrong.
- If failures mention missing MPS/data files, follow the dataset setup from `CONTRIBUTING.md`; do not report missing datasets as the code result.

## Output Format

Lead with findings:

```text
Finding: <root cause or strongest hypothesis>
Evidence:
- <job/run/test/log line>
- <code path or diff path>
Impact: <which jobs/users are affected>
Fix: <minimal change direction>
Verify:
- <exact local command>
- <exact CI job or test to rerun>
```

If confidence is low, say what is known, what is unknown, and the next command that would reduce uncertainty. Avoid presenting speculation as fact.
