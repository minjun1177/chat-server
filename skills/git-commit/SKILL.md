---
name: git-commit
description: >
  Write a Conventional Commits message from the current staged or unstaged
  changes. Use when the user asks to commit, asks for a commit message, or
  says "커밋", "커밋 메시지", "commit message".
allowed-tools: git_status, git_diff, run_cmd
---

# Writing a commit message

## Steps

1. Call `git_status` to see what is staged.
2. Call `git_diff` to read the actual change. Never write a message from the
   file names alone.
3. Pick the type from `types.md` bundled with this skill (read it if unsure).
4. Draft the message, then show it to the user before running any git command.
5. Only run `git commit` through `run_cmd` after the user approves the wording.

## Format

```
<type>(<scope>): <subject>

<body>
```

- **subject** — imperative mood, lower case, no trailing period, 50 chars or less.
  "add retry to upload", not "added retry" or "Adds retry.".
- **scope** — optional, the module or area touched (`auth`, `parser`, `tui`).
- **body** — optional, wrap at 72 chars. Explain *why*, not *what*; the diff
  already says what changed. Skip the body for a one-line, obvious change.

## Rules

- One logical change per commit. If the diff mixes a fix and a refactor, say so
  and propose splitting it rather than writing one vague message.
- Never write filler subjects: "update code", "fix bug", "various changes".
- If nothing is staged, say so and ask whether to stage everything first,
  instead of committing silently.
