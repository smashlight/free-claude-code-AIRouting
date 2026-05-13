<div align="center">

# 🤖 Free Claude Code + AUTO_ROUTE

**Fork** of [Alishahryar1/free-claude-code](https://github.com/Alishahryar1/free-claude-code) with **automatic model routing by task complexity**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![GitHub smashlight](https://img.shields.io/badge/GitHub-smashlight-181717?style=for-the-badge&logo=github)](https://github.com/smashlight/free-claude-code-AIRouting)

> 🔗 **Оригинальный проект:** https://github.com/Alishahryar1/free-claude-code

</div>

---

## 🔥 Что это и зачем

Это прокси-сервер для Claude Code CLI. Он сидит между Claude Code и AI-моделями, и **автоматически выбирает модель под задачу**:

| Если задача | Пойдёт на | Стоимость |
|---|---|---|
| ❓ Простой вопрос / чат | **OpenRouter** (бесплатные модели) | 💸 `$0` |
| ⚡ Рефакторинг / кодогенерация | **DeepSeek V4 Flash** | ⚡ дёшево |
| 💪 Сложная архитектура / алгоритмы | **DeepSeek V4 Pro** | 💪 мощно |

Больше не нужно думать, какую модель выбрать. Просто работаешь, а прокси сам решает.

---

## 🚀 Быстрая установка (2 минуты)

### 1. Установи Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

### 2. Установи uv (менеджер пакетов)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Склонируй и установи прокси

```bash
git clone https://github.com/smashlight/free-claude-code-AIRouting.git
cd free-claude-code-AIRouting
uv tool install --force .
```

### 4. Настрой API-ключи

Создай файл `~/.config/free-claude-code/.env`:

```bash
mkdir -p ~/.config/free-claude-code
```

Скопируй туда:

```ini
# ==== API Ключи ====
# DeepSeek (нужен для классификации задач + моделей flash/pro)
DEEPSEEK_API_KEY="sk-ваш-ключ-deepseek"

# OpenRouter (нужен для бесплатных моделей на простых задачах)
OPENROUTER_API_KEY="sk-or-v1-ваш-ключ-openrouter"

# ==== Модели (можно менять под себя) ====
MODEL_OPUS="deepseek/deepseek-v4-pro"           # Самая мощная
MODEL_SONNET="deepseek/deepseek-v4-flash"        # Средняя
MODEL_HAIKU="open_router/openrouter/free"        # Бесплатная
MODEL="deepseek/deepseek-v4-pro"                 # Запасная (fallback)

# ==== AUTO_ROUTE ====
AUTO_ROUTE_ENABLED=true
AUTO_ROUTE_CLASSIFIER_MODEL=deepseek/deepseek-v4-flash

# ==== Безопасность ====
ANTHROPIC_AUTH_TOKEN="freecc"

# ==== Отключение thinking (совместимость с DeepSeek) ====
ENABLE_MODEL_THINKING=false

# ==== Оптимизации ====
ENABLE_NETWORK_PROBE_MOCK=true
ENABLE_TITLE_GENERATION_SKIP=true
ENABLE_SUGGESTION_MODE_SKIP=true
ENABLE_FILEPATH_EXTRACTION_MOCK=true
```

### 5. Запусти прокси

В одном терминале:

```bash
cd free-claude-code-AIRouting
uv run python server.py
```

### 6. Запусти Claude Code

В другом терминале:

```bash
claude
```

Готово! 🎉 Теперь каждый твой запрос будет автоматически направляться на оптимальную модель.

---

## 🤖 Альтернатива: установка через AI

Скопируй этот README и отправь любой нейронке с промптом:

> Прочитай этот README и выполни все шаги по установке по порядку. Отвечай только когда будет нужно что-то уточнить или когда всё установится.

Она сама всё сделает шаг за шагом.

---

## ⚙️ Как это работает

### Архитектура

```
                          ┌─ SIMPLE        → OpenRouter (free)
Ваш запрос → Прокси ──────┼─ COMPLEX       → DeepSeek Flash
                          └─ VERY_COMPLEX   → DeepSeek Pro
                                ↑
                        Классификатор
                        (DeepSeek Flash,
                         ~1.5-2 сек)
```

Перед каждым запросом прокси делает быстрый классификационный вызов к DeepSeek Flash (стоит копейки, ~2 секунды). Классификатор смотрит на текст задачи и определяет её сложность. Затем запрос направляется на соответствующую модель.

### Промпт классификатора

Классификатор получает такой промпт:

```
Classify this coding task complexity.
- SIMPLE: Chat, questions, file reads, simple search, one-line fixes
- COMPLEX: Multi-file refactoring, architecture changes, code generation
- VERY_COMPLEX: System design, complex algorithms, performance optimization

Task: {текст задачи пользователя}
Response (one word):
```

---

## 🔧 Как поменять модели

В `.env` есть три тира:

```ini
# Тир Haiku — для самых простых задач
MODEL_HAIKU="open_router/openrouter/free"

# Тир Sonnet — для средних задач
MODEL_SONNET="deepseek/deepseek-v4-flash"

# Тир Opus — для сложных задач
MODEL_OPUS="deepseek/deepseek-v4-pro"

# Запасной вариант (если модель не определилась)
MODEL="deepseek/deepseek-v4-pro"
```

Ты можешь поставить любые модели из支持的 провайдеров:

| Провайдер | Формат | Пример |
|---|---|---|
| DeepSeek | `deepseek/имя-модели` | `deepseek/deepseek-v4-flash` |
| OpenRouter | `open_router/имя-модели` | `open_router/anthropic/claude-sonnet-4` |
| NVIDIA NIM | `nvidia_nim/модель` | `nvidia_nim/meta/llama-3.3-70b-instruct` |
| LM Studio | `lmstudio/модель` | `lmstudio/qwen2.5-7b` |
| Ollama | `ollama/модель` | `ollama/llama3.1` |

---

## 📋 Что нужно для работы

| Что | Где взять | Зачем |
|---|---|---|
| **DeepSeek API ключ** | https://platform.deepseek.com/api_keys | Классификация задач + модели flash/pro |
| **OpenRouter API ключ** | https://openrouter.ai/keys | Бесплатные модели для простых задач |
| **Claude Code CLI** | `npm install -g @anthropic-ai/claude-code` | Интерфейс для работы |

---

## 💡 Примеры работы

| Твой запрос | Сложность | Куда идёт |
|---|---|---|
| "What is 2+2?" | SIMPLE | OpenRouter free (бесплатно) |
| "Write a bash script that finds large files" | COMPLEX | DeepSeek Flash |
| "Design a distributed cache system" | VERY_COMPLEX | DeepSeek Pro |
| "Refactor auth module to JWT" | COMPLEX → VERY_COMPLEX | DeepSeek Flash → Pro |

---

## 📝 Команды для быстрого запуска

```bash
# Запустить прокси-сервер
cd ~/free-claude-code-AIRouting && uv run python server.py

# Запустить Claude Code (в другом терминале)
claude
```

---

## 🔗 Ссылки

- **[Оригинальный проект](https://github.com/Alishahryar1/free-claude-code)** — бесплатный Claude Code proxy
- **[DeepSeek](https://platform.deepseek.com/)** — API ключ для DeepSeek
- **[OpenRouter](https://openrouter.ai/)** — API ключ для OpenRouter
- **[DeepSeek-TUI](https://github.com/Hmbown/DeepSeek-TUI)** — идея Auto Mode роутинга

---

<div align="center">
Made with ❤️ by <a href="https://github.com/smashlight">smashlight</a>
</div>
