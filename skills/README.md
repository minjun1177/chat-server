# Skills

A skill is an instruction pack the model loads on demand. Only each skill's
`name` and `description` sit in the system prompt; the body is pulled in only
when the model calls `use_skill` (or you run `/skill <name>`), so a big library
costs almost nothing until it is actually used.

## Layout

```
skills/
  my-skill/
    SKILL.md          <- required
    reference.md      <- optional bundled files, any name
    helper.py
  quick-skill.md      <- one-file skill, no bundled files
```

Skills are searched in this order, and the first match on a name wins:

| Source    | Path                          |
| :-------- | :---------------------------- |
| `project` | `./skills/`                   |
| `user`    | `~/.localchat/skills/`        |

## SKILL.md

```markdown
---
name: git-commit
description: Use when the user asks for a commit message. Triggers on "commit", "커밋".
allowed-tools: run_cmd, read_file
---

Instructions in plain markdown go here.
```

- `name` — optional; the folder (or file) name is used when it is missing.
- `description` — the only thing the model sees before loading. Write it as
  *when to use this*, and include the words a user would actually type.
- `allowed-tools` — optional, comma-separated or a YAML list. Advisory: it is
  shown to the model as the tool set the skill expects, not enforced by the
  harness.

Block scalars work too:

```markdown
description: >
  A long description folded
  onto one line.
```

Files bundled next to `SKILL.md` are listed with absolute paths when the skill
loads, so the model can open them with `read_file` or run them with `run_cmd`.

## Commands

| Command           | Description                                        |
| :---------------- | :------------------------------------------------- |
| `/skills`         | List discovered skills                             |
| `/skills reload`  | Rescan the directories and refresh the system prompt |
| `/skill <name>`   | Load a skill into the conversation by hand         |
