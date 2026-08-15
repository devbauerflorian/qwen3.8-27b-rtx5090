# Contributing

## Conventional Commits

This repository follows [Conventional Commits](https://www.conventionalcommits.org/).
Write the subject line as:

```
<type>[optional scope]: <short imperative subject (max 72 chars)>
```

### Types

| Type       | Use for |
|------------|---------|
| `feat`     | New functionality (e.g. new compose flags, new model variants) |
| `fix`      | Bug fixes |
| `docs`     | Documentation only (README, comments) |
| `refactor` | Changes that neither fix a bug nor add a feature |
| `perf`     | Performance-oriented changes |
| `build`    | Build system or dependencies (Dockerfile, pinned versions) |
| `ci`       | CI configuration |
| `chore`    | Maintenance (cleanup, dependency bumps, tooling) |
| `revert`   | Reverting a previous commit |

A scope in parentheses is optional, e.g. `feat(model): support 5090 D variant`.
A breaking change gets a `!` after the type and a `BREAKING CHANGE:` footer.

### Rules of thumb

- Imperative/infinitive ("add", "fix", "update" — not "added"/"fixes"), no trailing dot
- Subject ≤ 72 characters; keep the body for *why*, reference issues as `#123`
- One logical change per commit

Examples:

```
feat: serve qwen3.8-30b-a3b on second GPU
fix(compose): set VLLM_API_KEY only when provided in .env
build: pin vllm to 0.25.4 for reproducible images
docs: document .env setup for multiple hosts
```

### Optional: commit template

A template lives at `.gitmessage`. Apply it once per machine:

```bash
git config commit.template .gitmessage
```
