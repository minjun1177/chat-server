# Conventional Commit types

| Type       | Use for                                                        |
| :--------- | :------------------------------------------------------------- |
| `feat`     | A new capability the user can see                               |
| `fix`      | A bug fix                                                       |
| `docs`     | Documentation only                                              |
| `style`    | Formatting, whitespace, no behaviour change                     |
| `refactor` | Restructuring that neither fixes a bug nor adds a feature       |
| `perf`     | A change made specifically to improve performance               |
| `test`     | Adding or correcting tests                                      |
| `build`    | Build system, packaging, or dependency changes                  |
| `ci`       | CI configuration and scripts                                    |
| `chore`    | Housekeeping that touches no source (e.g. `.gitignore`)         |
| `revert`   | Reverting an earlier commit; name it in the body                |

## Breaking changes

Add `!` after the type/scope and explain it in a footer:

```
feat(api)!: drop support for v1 tokens

BREAKING CHANGE: clients must re-authenticate with a v2 token.
```
