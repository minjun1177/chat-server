---
name: code-review
description: >
  Review a diff or a source file for correctness bugs and cleanups. Use when
  the user asks for a code review, asks "이 코드 리뷰해줘", "리뷰", "review this",
  or asks what is wrong with a change they just made.
allowed-tools: git_diff, read_file, get_code_skeleton, search_in_file
---

# Reviewing code

## Steps

1. Get the change: `git_diff` for uncommitted work, `read_file` for a specific
   file. For a large unfamiliar file, start with `get_code_skeleton` to see the
   structure before reading the body.
2. Read the surrounding code, not just the changed lines. A call site three
   functions away is often where the bug actually shows up; use
   `search_in_file` to find callers of anything the change touches.
3. Report findings ordered by severity, most severe first.

## What counts as a finding

Report it only if you can name a concrete failure: the input or state that
triggers it, and the wrong result that follows.

- Correctness: off-by-one, wrong operator, unhandled `None`/empty case,
  swapped arguments, a condition that can never be true.
- Resource and state: files or connections left open, mutable default
  arguments, shared state mutated from more than one place.
- Error handling: a bare `except` that swallows the error that matters, an
  error path that returns a value the caller reads as success.
- Reuse: the repo already has a helper doing exactly this. Name the helper and
  its file.
- Simplification: the same result in materially less code.

## What to leave alone

Style the surrounding code already uses, naming preferences, missing type
hints, and anything you would phrase as "consider maybe". If the change is
clean, say it is clean — do not pad the report to look thorough.

## Output

For each finding:

```
<file>:<line> — <one sentence: the defect>
Fails when: <the input or state, and the wrong result>
Fix: <the change, in a line or two>
```
