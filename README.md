<div align="center">

# 🤖 Free Claude Code + AUTO_ROUTE

**Fork** of [Alishahryar1/free-claude-code](https://github.com/Alishahryar1/free-claude-code) with **automatic model routing by task complexity**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![GitHub smashlight](https://img.shields.io/badge/GitHub-smashlight-181717?style=for-the-badge&logo=github)](https://github.com/smashlight/free-claude-code-AIRouting)

> 🔗 **Original project:** https://github.com/Alishahryar1/free-claude-code

</div>

---

<p align="center">
  <strong>🇬🇧 English</strong> |
  <a href="./README.ru.md">🇷🇺 Русский</a>
</p>

## 🔥 What it is and why it exists

This is a proxy server for Claude Code CLI. It sits between Claude Code and AI models, and **automatically chooses the right model for the task**:

| If the task is | It goes to | Cost |
|---|---|---|
| ❓ Simple question / chat | **OpenRouter** (free models) | 💸 `$0` |
| ⚡ Refactoring / code generation | **DeepSeek V4 Flash** | ⚡ cheap |
| 💪 Complex architecture / algorithms | **DeepSeek V4 Pro** | 💪 powerful |

You no longer need to think about which model to use. Just work — the proxy decides for you.

---

## 🚀 Quick setup (2 minutes)

### 1. Install Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

### 2. Install uv (package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Clone and install the proxy

```bash
git clone https://github.com/smashlight/free-claude-code-AIRouting.git
cd free-claude-code-AIRouting
uv tool install --force .
```

### 4. Configure API keys

Create `~/.config/free-claude-code/.env`:

```bash
mkdir -p ~/.config/free-claude-code
```

Copy this into it:

```ini
# ==== API keys ====
# DeepSeek (required for task classification + flash/pro models)
DEEPSEEK_API_KEY="sk-your-deepseek-key"

# OpenRouter (required for free models on simple tasks)
OPENROUTER_API_KEY="sk-or-v1-your-openrouter-key"

# ==== Models (you can change these) ====
MODEL_OPUS="deepseek/deepseek-v4-pro"           # Most powerful
MODEL_SONNET="deepseek/deepseek-v4-flash"        # Medium tier
MODEL_HAIKU="open_router/openrouter/free"        # Free tier
MODEL="deepseek/deepseek-v4-pro"                 # Fallback

# ==== AUTO_ROUTE ====
AUTO_ROUTE_ENABLED=true
AUTO_ROUTE_CLASSIFIER_MODEL=deepseek/deepseek-v4-flash

# ==== Security ====
ANTHROPIC_AUTH_TOKEN="freecc"

# ==== Thinking ====
# DeepSeek V4 Flash/Pro support thinking; OpenRouter free tier stays off.
ENABLE_MODEL_THINKING=false
ENABLE_HAIKU_THINKING=false
ENABLE_SONNET_THINKING=true
ENABLE_OPUS_THINKING=true

# ==== Optimizations ====
ENABLE_NETWORK_PROBE_MOCK=true
ENABLE_TITLE_GENERATION_SKIP=true
ENABLE_SUGGESTION_MODE_SKIP=true
ENABLE_FILEPATH_EXTRACTION_MOCK=true
```

### 5. Start the proxy

In one terminal:

```bash
cd free-claude-code-AIRouting
uv run python server.py
```

### 6. Create launch aliases

The proxy exposes `fcc-claude`. Add this to `~/.zshrc` (or `~/.bashrc`):

```bash
alias claudeds='fcc-claude'
alias claudeds-yolo='fcc-claude -- --dangerously-skip-permissions'
```

Apply it:

```bash
source ~/.zshrc
```

### 7. Run Claude Code through the proxy

In one terminal, start the proxy:

```bash
cd free-claude-code-AIRouting
uv run python server.py
```

In another terminal, start Claude Code:

```bash
claudeds
```

Done! 🎉 Every request is now automatically routed to the best-fit model.

> **Important:** the regular `claude` command runs Claude Code directly against Anthropic (official models, paid). Use `claudeds` to route requests through this proxy with AUTO_ROUTE.

---

## 🤖 AI-assisted setup (for Windows users)

If you are on Windows or want to automate the setup, give this prompt to an AI assistant (Claude Code, ChatGPT, etc.):

```
Clone https://github.com/smashlight/free-claude-code-AIRouting.git into a project folder.
Read the README and complete all installation and proxy configuration steps.
Connect it to DeepSeek API using my key: "sk-your-deepseek-key"

IMPORTANT — model names: .env.example contains model names, but DeepSeek may rename models occasionally.
Before starting the proxy, query the DeepSeek API (GET https://api.deepseek.com/models with the API key),
check the current model names (likely deepseek-v4-flash and deepseek-v4-pro), and put them into .env.

Create two aliases:
- claudeds — run Claude Code through the proxy
- claudeds-yolo — same, plus --dangerously-skip-permissions
```

The assistant can clone the repo, install dependencies, configure `.env`, check current model names, and create aliases for you.

---

## ⚙️ How it works

### Architecture

```text
                          ┌─ SIMPLE        → OpenRouter (free)
Your request → Proxy ─────┼─ COMPLEX       → DeepSeek Flash
                          └─ VERY_COMPLEX  → DeepSeek Pro
                                ↑
                          Classifier
                         (DeepSeek Flash,
                          ~1.5–2 sec)
```

Before each request, the proxy makes a fast classification call to DeepSeek Flash (very cheap, about 2 seconds). The classifier looks at the task text and decides its complexity. Then the request is routed to the corresponding model.

### Classifier prompt

The classifier receives a prompt like this:

```text
Classify this coding task complexity.
- SIMPLE: Chat, questions, file reads, simple search, one-line fixes
- COMPLEX: Multi-file refactoring, architecture changes, code generation
- VERY_COMPLEX: System design, complex algorithms, performance optimization

Task: {user task text}
Response (one word):
```

---

## 🔧 Changing models

The default AUTO_ROUTE setup keeps `MODEL_HAIKU` on OpenRouter/free without thinking, while `MODEL_SONNET` (`deepseek-v4-flash`) and `MODEL_OPUS` (`deepseek-v4-pro`) run with thinking enabled. DeepSeek thinking with tool calls depends on replaying prior `reasoning_content`/thinking blocks in follow-up requests; the DeepSeek adapter preserves that replay and strips unsupported redacted thinking.

The `.env` file has three tiers:

```ini
# Haiku tier — for the simplest tasks
MODEL_HAIKU="open_router/openrouter/free"

# Sonnet tier — for medium tasks
MODEL_SONNET="deepseek/deepseek-v4-flash"

# Opus tier — for complex tasks
MODEL_OPUS="deepseek/deepseek-v4-pro"

# Fallback if the model cannot be resolved
MODEL="deepseek/deepseek-v4-pro"
```

You can use any model from supported providers:

| Provider | Format | Example |
|---|---|---|
| DeepSeek | `deepseek/model-name` | `deepseek/deepseek-v4-flash` |
| OpenRouter | `open_router/model-name` | `open_router/anthropic/claude-sonnet-4` |
| NVIDIA NIM | `nvidia_nim/model` | `nvidia_nim/meta/llama-3.3-70b-instruct` |
| LM Studio | `lmstudio/model` | `lmstudio/qwen2.5-7b` |
| Ollama | `ollama/model` | `ollama/llama3.1` |

---

## 📋 Requirements

| What | Where to get it | Why |
|---|---|---|
| **DeepSeek API key** | https://platform.deepseek.com/api_keys | Task classification + flash/pro models |
| **OpenRouter API key** | https://openrouter.ai/keys | Free models for simple tasks |
| **Claude Code CLI** | `npm install -g @anthropic-ai/claude-code` | Main coding interface |

---

## 💡 Examples

| Your request | Complexity | Routed to |
|---|---|---|
| "What is 2+2?" | SIMPLE | OpenRouter free (free) |
| "Write a bash script that finds large files" | COMPLEX | DeepSeek Flash |
| "Design a distributed cache system" | VERY_COMPLEX | DeepSeek Pro |
| "Refactor auth module to JWT" | COMPLEX → VERY_COMPLEX | DeepSeek Flash → Pro |

---

## 📝 Quick launch commands

```bash
# Start the proxy server
cd ~/free-claude-code-AIRouting && uv run python server.py

# Start Claude Code through the proxy (in another terminal)
claudeds

# Fast launch with auto-confirmation
claudeds-yolo
```

---

## 🔗 Links

- **[Original project](https://github.com/Alishahryar1/free-claude-code)** — free Claude Code proxy
- **[DeepSeek](https://platform.deepseek.com/)** — DeepSeek API key
- **[OpenRouter](https://openrouter.ai/)** — OpenRouter API key
- **[DeepSeek-TUI](https://github.com/Hmbown/DeepSeek-TUI)** — Auto Mode routing inspiration

---

<div align="center">
Made with ❤️ by <a href="https://github.com/smashlight">smashlight</a>
</div>
