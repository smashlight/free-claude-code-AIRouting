# Plan: Enable DeepSeek Thinking for AUTO_ROUTE DeepSeek tiers

Date: 2026-05-13
Repo: `smashlight/free-claude-code-AIRouting`
Working branch proposal: `feat/deepseek-thinking-routing`

## Goal

Enable DeepSeek thinking mode for both DeepSeek-backed AUTO_ROUTE tiers while keeping OpenRouter/free simple-tier routing stable:

- `SIMPLE -> Haiku -> open_router/openrouter/free` without thinking
- `COMPLEX -> Sonnet -> deepseek/deepseek-v4-flash` with thinking
- `VERY_COMPLEX -> Opus -> deepseek/deepseek-v4-pro` with thinking

DeepSeek documentation confirms that both `deepseek-v4-flash` and `deepseek-v4-pro` support dual modes: Thinking and Non-Thinking.

## Why this needs care

DeepSeek thinking mode is not only a config toggle. The risky area is tool-call compatibility.

According to DeepSeek docs:

- Thinking mode returns reasoning via `reasoning_content`.
- If a model performs tool calls while in thinking mode, the generated `reasoning_content` must be passed back in subsequent requests.
- If this is not handled correctly, DeepSeek API can return `400` errors.

So the implementation should first verify the adapter behavior, then enable config defaults.

## Branch workflow

Do not work directly on `main`.

1. Start from latest `origin/main`.
2. Create a dedicated branch:

   ```bash
   git checkout main
   git pull --ff-only origin main
   git checkout -b feat/deepseek-thinking-routing
   ```

3. Commit all thinking-mode changes to this branch.
4. Push branch to origin.
5. Test thoroughly.
6. Only after green checks and manual review, merge into `main`.

Suggested final merge approach:

```bash
git checkout main
git pull --ff-only origin main
git merge --no-ff feat/deepseek-thinking-routing
git push origin main
```

Alternative: open a PR from `feat/deepseek-thinking-routing` into `main` and merge on GitHub.

## Implementation plan

### 1. Audit current thinking flow

Inspect these files first:

- `api/services.py`
- `providers/deepseek/request.py`
- `config/settings.py`
- `tests/api/test_api.py`
- `tests/config/test_config.py`
- any existing thinking-related tests

Questions to answer:

- Where does Claude-style `thinking` enter request processing?
- How are tier-specific flags applied?
- How is `ENABLE_MODEL_THINKING` combined with `ENABLE_HAIKU_THINKING`, `ENABLE_SONNET_THINKING`, and `ENABLE_OPUS_THINKING`?
- Does DeepSeek adapter convert Claude/Anthropic thinking into DeepSeek-compatible format?
- Does the adapter preserve `reasoning_content` for tool-call turns?
- Are there code paths that intentionally disable thinking for safety?
- Are those safety paths still necessary after the desired change?

### 2. Confirm target config behavior

Desired defaults for `.env.example`:

```ini
ENABLE_MODEL_THINKING=false
ENABLE_HAIKU_THINKING=false
ENABLE_SONNET_THINKING=true
ENABLE_OPUS_THINKING=true
```

Meaning:

- Global default stays conservative.
- OpenRouter/free simple tier stays non-thinking.
- DeepSeek Flash/Sonnet tier uses thinking.
- DeepSeek Pro/Opus tier uses thinking.

### 3. Implement adapter/config changes if needed

Depending on audit results:

- Ensure requests routed to `MODEL_SONNET=deepseek/deepseek-v4-flash` can receive thinking.
- Ensure requests routed to `MODEL_OPUS=deepseek/deepseek-v4-pro` can receive thinking.
- Ensure requests routed to `MODEL_HAIKU=open_router/openrouter/free` do not receive thinking.
- If DeepSeek tool-call thinking requires special handling, preserve/pass `reasoning_content` correctly.
- If current code disables thinking on tool follow-ups, decide whether to:
  - keep that safety fallback, or
  - replace it with correct `reasoning_content` propagation.

Preference: correctness over forcing thinking everywhere. If tool-call thinking is not safely supported yet, document the limitation and keep a guarded fallback.

### 4. Tests to add or update

Add focused tests for:

1. `COMPLEX` / Sonnet tier:
   - selected model: `deepseek/deepseek-v4-flash`
   - thinking enabled

2. `VERY_COMPLEX` / Opus tier:
   - selected model: `deepseek/deepseek-v4-pro`
   - thinking enabled

3. `SIMPLE` / Haiku tier:
   - selected model: `open_router/openrouter/free`
   - thinking disabled / stripped

4. Tool-call follow-up path:
   - thinking mode does not produce an invalid DeepSeek request
   - `reasoning_content` is preserved if required by the adapter design
   - no unsafe `redacted_thinking` or unsupported Claude-only blocks are sent to DeepSeek

5. Config validation:
   - new thinking defaults are represented in `.env.example`
   - settings load expected tier flags

### 5. Documentation updates

Update at least:

- `.env.example`
- `README.md`
- `README.ru.md`
- possibly `docs/plans/2026-05-13-auto-route.md` if it describes thinking/config defaults

Docs should state:

- DeepSeek V4 Flash and DeepSeek V4 Pro support Thinking / Non-Thinking.
- AUTO_ROUTE enables thinking on DeepSeek-backed tiers by default.
- OpenRouter/free simple tier keeps thinking disabled for compatibility.
- Tool-call thinking requires careful handling of `reasoning_content`.

### 6. Verification gates

Before considering branch ready:

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

If possible, also do a live/manual smoke test with local proxy:

1. Start server:

   ```bash
   uv run python server.py
   ```

2. Send representative tasks through Claude Code/proxy:

   - simple question -> OpenRouter/free, no thinking
   - medium coding task -> DeepSeek Flash, thinking enabled
   - complex architecture task -> DeepSeek Pro, thinking enabled
   - tool-using task -> no DeepSeek 400 error

### 7. Commit strategy

Use one or two clean commits.

Option A, one commit:

```text
feat(config): enable DeepSeek thinking for routed tiers
```

Commit body should mention:

- Enables thinking for Sonnet/Opus DeepSeek tiers.
- Keeps Haiku/OpenRouter free tier non-thinking.
- Updates docs/env examples.
- Adds tests for tier-specific thinking and tool-call compatibility.

Option B, two commits:

```text
test(api): cover DeepSeek thinking routing behavior
feat(config): enable DeepSeek thinking for routed tiers
```

Prefer Option B if the code changes are non-trivial.

### 8. Merge criteria

Merge branch only when:

- lint/type/tests are green;
- README and `.env.example` are synchronized;
- tool-call thinking behavior is either safely supported or explicitly guarded/documented;
- commit history is clean and descriptive.

## Starter prompt for a new session

```text
We are working on my fork: https://github.com/smashlight/free-claude-code-AIRouting
Local repo path may already exist at: /tmp/free-claude-code-AIRouting-review/repo

Goal: create a dedicated branch and safely enable DeepSeek thinking for AUTO_ROUTE DeepSeek tiers.

Important routing mapping:
- SIMPLE -> Haiku -> open_router/openrouter/free
- COMPLEX -> Sonnet -> deepseek/deepseek-v4-flash
- VERY_COMPLEX -> Opus -> deepseek/deepseek-v4-pro

DeepSeek docs confirm both deepseek-v4-flash and deepseek-v4-pro support Thinking / Non-Thinking modes.
Target default config:
ENABLE_MODEL_THINKING=false
ENABLE_HAIKU_THINKING=false
ENABLE_SONNET_THINKING=true
ENABLE_OPUS_THINKING=true

Important caution:
DeepSeek thinking + tool calls requires reasoning_content to be preserved/passed back correctly. If not handled correctly, DeepSeek can return 400. So first audit the adapter before blindly enabling flags.

Please follow the saved plan in:
docs/plans/2026-05-13-deepseek-thinking.md

Workflow:
1. Start from latest origin/main.
2. Create branch feat/deepseek-thinking-routing.
3. Audit api/services.py, providers/deepseek/request.py, config/settings.py, tests.
4. Implement the safest fix.
5. Update .env.example, README.md, README.ru.md, and docs if needed.
6. Add/update tests for Sonnet/Flash thinking, Opus/Pro thinking, Haiku/OpenRouter no-thinking, and tool-call compatibility.
7. Run:
   uv run ruff check .
   uv run ty check
   uv run pytest
8. Commit with a descriptive message and push the branch, but do not merge into main until we review.

Use GitHub auth via gh if needed. Do not ask me to paste tokens in chat.
```
