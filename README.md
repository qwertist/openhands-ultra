# 🤖 OpenHands Ultra

<div align="center">

<img src="assets/logo.png" alt="OpenHands Ultra" width="200">

**Autonomous AI Coding Agent with Git-Native State Management**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Quick Start](#-quick-start) • [Features](#-features) • [Commands](#-commands) • [Architecture](#-architecture) • [Configuration](#%EF%B8%8F-configuration)

</div>

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/qwertist/openhands-ultra.git
cd openhands-ultra

# Configure API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# Run
python3 openhands.py
```

First run automatically:
- Creates `.env` from template
- Installs dependencies (`textual`, `sentence-transformers`)
- Configures LLM templates

---

## ✨ Features

### 🎯 Autonomous Coding
- **Ralph Daemon** — AI agent runs independently inside Docker
- **Task Planning** — Automatically breaks projects into tasks
- **Self-Healing** — Stuck detection with recovery strategies
- **Architect Reviews** — Periodic code quality checks

### 📦 Git-Native State
- **Tasks in Git** — Stored as git blob, not files
- **Full History** — `git reflog` for all state changes
- **Crash Recovery** — State survives any failure
- **No File Conflicts** — Atomic operations via git refs

### 🔒 Security Hardened
- **50+ Security Fixes** — Shell injection, path traversal, ReDoS
- **Input Validation** — All user inputs sanitized
- **Credential Protection** — Secrets redacted from logs
- **Race Condition Free** — Proper locking throughout

### 🧠 Smart Context
- **200K+ Tokens** — Optimized for large context models
- **Hierarchical Memory** — Hot/warm/cold tiers
- **Semantic Search** — Find relevant code instantly
- **Context Condensing** — Automatic summarization

---

## 📋 Commands

### Launch TUI
```bash
python3 openhands.py
```

### Quick Start Project
```bash
python3 openhands.py myproject
```

### CLI Options
```bash
python3 openhands.py --help           # Show help
python3 openhands.py --version        # Show version
python3 openhands.py --list           # List all projects
python3 openhands.py --setup          # Re-run initial setup
python3 openhands.py myproject        # Quick-start specific project
```

### TUI Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `n` | New Project |
| `s` | Start Session |
| `r` | Start/Monitor Ralph |
| `p` | Project Settings |
| `c` | Container Management |
| `l` | View Logs |
| `F5` | Refresh |

### Ralph Monitor Controls

| Key | Action |
|-----|--------|
| `p` | Pause/Resume |
| `s` | Stop Daemon |
| `l` | View Iteration Logs |
| `t` | View Tasks |
| `Esc` | Back (daemon keeps running) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  HOST                                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  openhands.py (TUI)                                    │ │
│  │  ├── Project Management                                │ │
│  │  ├── Container Control (Docker API)                    │ │
│  │  ├── Ralph Lifecycle (start/stop/monitor)              │ │
│  │  └── Git-Native State (TaskManager, GitStateManager)   │ │
│  └────────────────────────────────────────────────────────┘ │
│            │                                                 │
│            │ docker exec                                     │
│            ▼                                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  DOCKER CONTAINER                                       │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  ralph_daemon.py                                  │  │ │
│  │  │  ├── Autonomous iteration loop                   │  │ │
│  │  │  ├── HierarchicalMemory (context management)     │  │ │
│  │  │  ├── SemanticSearch (code understanding)         │  │ │
│  │  │  ├── StuckDetector (recovery strategies)         │  │ │
│  │  │  └── CircuitBreaker (failure protection)         │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │           │                                             │ │
│  │           ▼                                             │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  OpenHands Agent (per iteration)                 │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Git-Native State Storage

```bash
# Tasks (as git blob)
git show refs/ralph/tasks

# Current iteration
cat .git/ralph/iteration

# Current task
cat .git/ralph/task

# Daemon status
cat .git/ralph/status

# Checkpoints (as git tags)
git tag -l "ralph/cp/*"

# Learnings (as git notes)
git log --show-notes=learnings

# History of any state
git reflog refs/ralph/tasks
```

---

## 📁 Project Structure

```
openhands-ultra/
├── openhands.py              # Main TUI application
├── .env                      # API keys (create from .env.example)
├── .env.example              # Template for .env
├── templates/
│   ├── llm/                  # LLM configurations
│   │   ├── claude-sonnet-4/
│   │   ├── claude-opus-4.5/
│   │   └── kimi-k2/
│   ├── mcp/                  # MCP server configs
│   └── tools/ralph/
│       ├── ralph_daemon.py   # Autonomous daemon
│       └── git_state.py      # State management
├── formulas/                 # Workflow templates (TOML)
│   ├── bugfix.toml
│   ├── feature.toml
│   └── refactor.toml
└── projects/                 # Your projects (gitignored)
    └── myproject/
        ├── workspace/        # Code (mounted in container)
        │   └── .ralph/       # Runtime config only
        ├── config/           # LLM settings
        └── data/             # Persistent data
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
OPENAI_API_KEY=sk-...
KIMI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### LLM Configuration

Edit `templates/llm/<model>/agent_settings.json`:

```json
{
  "llm": {
    "model": "anthropic/claude-sonnet-4-20250514",
    "api_key": "${ANTHROPIC_API_KEY}",
    "max_tokens": 8192
  },
  "agent": {
    "type": "CodeActAgent"
  }
}
```

### Ralph Settings

When starting Ralph, configure:

| Setting | Default | Description |
|---------|---------|-------------|
| Max Iterations | 0 | Limit iterations (0 = unlimited) |
| Architect Interval | 10 | Code review every N iterations |
| Condense Interval | 15 | Context summarization frequency |

---

## 🍳 Formulas

Formulas are TOML templates for reusable workflows.

### Use a Formula

```bash
# In Ralph session or TUI
ralph cook bugfix --var bug_description="Login fails"
ralph cook feature --var feature_name="Auth" --var feature_description="JWT auth"
```

### Built-in Formulas

**bugfix.toml** — Bug fix workflow
```
reproduce → fix → verify
```

**feature.toml** — New feature workflow
```
design → implement → test → document
```

**refactor.toml** — Refactoring workflow
```
analyze → test-before → refactor → verify
```

### Create Custom Formula

```toml
# formulas/my-workflow.toml
description = "My custom workflow"

[vars.target]
description = "What to work on"
required = true

[[steps]]
id = "step1"
title = "First step"
description = "Do {{target}}"

[[steps]]
id = "step2"
title = "Second step"
description = "Verify {{target}}"
needs = ["step1"]
```

---

## 🐳 Docker Management

### Container Commands

```bash
# List containers
docker ps -a | grep openhands

# Shell into container
docker exec -it openhands-myproject bash

# View daemon logs
docker exec openhands-myproject cat /workspace/.ralph/ralph_daemon.log

# Check daemon status
docker exec openhands-myproject pgrep -f ralph_daemon.py
```

### Manual Daemon Control

```bash
# Start daemon
docker exec openhands-myproject bash -c "
  cd /workspace
  setsid python3 -u /workspace/.ralph/ralph_daemon.py &
"

# Stop daemon
docker exec openhands-myproject pkill -f ralph_daemon.py

# View heartbeat
docker exec openhands-myproject cat /workspace/.ralph/heartbeat
```

---

## 🔧 Troubleshooting

### Common Issues

**"Docker not available"**
```bash
sudo systemctl start docker
# or on macOS: open -a Docker
```

**"Container won't start"**
```bash
docker pull docker.openhands.dev/openhands/runtime:latest-nikolaik
docker system prune -a  # Free space
```

**"Daemon keeps crashing"**
```bash
# Check logs
docker exec openhands-myproject cat /workspace/.ralph/ralph_daemon.log

# Check dependencies
docker exec openhands-myproject python3 -c "import sentence_transformers; print('OK')"
```

**"Tasks not syncing"**
```bash
# Check git state
docker exec openhands-myproject git show refs/ralph/tasks

# Force refresh
docker exec openhands-myproject git reflog refs/ralph/tasks
```

### Debug Commands

```bash
# Full system status
python3 openhands.py --list

# Container resources
docker stats openhands-myproject

# Daemon memory usage
docker exec openhands-myproject ps aux | grep ralph
```

---

## 📊 Monitoring

### Ralph Status

The TUI shows real-time status:
- Current iteration
- Active task
- Heartbeat age
- Memory usage
- Recent output

### Logs

```bash
# Daemon log (all output)
~/openhands/projects/myproject/workspace/.ralph/ralph_daemon.log

# Per-iteration logs
~/openhands/projects/myproject/workspace/.ralph/iterations/

# Watchdog log
~/openhands/projects/myproject/workspace/.ralph/watchdog.log
```

### Metrics

Ralph tracks:
- Iterations completed/failed
- Tasks done/pending
- Stuck recoveries
- Circuit breaker trips

---

## 🔒 Security

### Implemented Protections

- ✅ Shell injection prevention (shlex.quote, stdin passing)
- ✅ Path traversal blocking (resolve + relative_to)
- ✅ Symlink attack prevention
- ✅ ReDoS protection (line-by-line processing)
- ✅ Log injection sanitization
- ✅ Credential redaction in output
- ✅ PID file race condition fix (flock)
- ✅ Atomic file operations
- ✅ Input validation on all user data
- ✅ Command allowlists for MCP

### Best Practices

- Keep `.env` out of git (already in .gitignore)
- Use dedicated API keys per project
- Review daemon logs for anomalies
- Keep Docker images updated

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Make changes
4. Run tests (`python3 -m pytest tests/`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing`)
7. Open Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- [OpenHands](https://github.com/All-Hands-AI/OpenHands) — AI coding agent
- [Gastown](https://github.com/steveyegge/gastown) — Git-native state inspiration
- [Textual](https://github.com/Textualize/textual) — TUI framework

---

<div align="center">

**Built for autonomous coding** 🚀

</div>
