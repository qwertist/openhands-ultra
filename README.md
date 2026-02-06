# 🤖 OpenHands Max

<div align="center">

![Version](https://img.shields.io/badge/version-5.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

**Autonomous AI Coding Agent with Git-Native State Management**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [v5.0 Architecture](#-v50-git-native-architecture) • [Formulas](#-formulas)

</div>

---

## ✨ Features

### 🎯 Core Features
- **Autonomous Coding** — AI agent works independently on complex projects
- **Git-Native State** — All state stored in git (refs, tags, notes)
- **Bead-Style Task IDs** — Structured IDs like `oh-k7m2x` for reliable tracking
- **Formula System** — TOML templates for reusable workflows
- **100+ Security Fixes** — Hardened for production use

### 🖥️ Terminal User Interface
- **Project Management** — Create, configure, and manage AI coding projects
- **Container Control** — Start, stop, restart Docker containers
- **Session Management** — Background sessions that survive terminal close
- **Real-time Monitoring** — Watch agent progress with live output

### 🤖 Ralph Autonomous Daemon (v3.0)
- **Container-Native** — Runs inside Docker, survives TUI restarts
- **Smart Context** — 200K+ token support with hierarchical memory
- **Self-Healing** — Stuck detection with automatic recovery
- **Git Integration** — Commits, notes, and tags for full history

---

## 🆕 v5.0 Git-Native Architecture

### Before (v4.0) → After (v5.0)

| Aspect | v4.0 (Files) | v5.0 (Git-Native) |
|--------|--------------|-------------------|
| State | `state.json` | `.git/ralph/*` refs |
| Checkpoints | `checkpoint.json` | Git tags `ralph/cp/iter-N` |
| Learnings | `learnings/*.json` | Git notes `refs/notes/learnings` |
| Iterations | `iterations/*.json` | Commits `[Ralph:Iter:N]` |
| Handoffs | `handoff.json` | Git notes `refs/notes/handoff` |
| Tasks | `prd.json` (numbered) | `tasks.json` (bead IDs) |

### Files in `.ralph/` (Reduced from 10+ to 3)

```
workspace/.ralph/
├── config.json          # Runtime configuration
├── tasks.json           # Tasks with IDs (oh-xxxxx)
└── formulas/            # TOML workflow templates
    ├── bugfix.toml
    ├── feature.toml
    └── refactor.toml
```

### Git Storage

```bash
# State
.git/ralph/iteration    # Current iteration number
.git/ralph/task         # Current task ID  
.git/ralph/status       # running/paused/stopped

# Checkpoints (git tags)
git tag -l "ralph/cp/*"
git show ralph/cp/iter-42

# Iteration history
git log --grep="[Ralph:Iter:"

# Learnings (git notes)
git log --show-notes=learnings
```

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Docker (running)
- Git
- 4GB+ RAM recommended

### Quick Install

```bash
# Clone the repository
git clone https://github.com/qwertist/openhands-max.git
cd openhands-max

# Copy environment template
cp .env.example .env
# Edit .env with your API key

# Run (dependencies auto-install)
python3 openhands.py
```

Auto-installs:
- `textual` — TUI framework
- `sentence-transformers` — Semantic search (~500MB)

---

## 🚀 Quick Start

### Launch TUI
```bash
python3 openhands.py
```

### Quick Start Project
```bash
python3 openhands.py myproject
```

### Using Formulas

```bash
# Create tasks from a formula
# (Inside Ralph session or via TUI)

# Bug fix workflow: reproduce → fix → verify
ralph cook bugfix --var bug_description="Login button doesn't work"

# Feature workflow: design → implement → test → document  
ralph cook feature --var feature_name="User Auth" --var feature_description="JWT-based authentication"

# Refactor workflow: analyze → test-before → refactor → verify
ralph cook refactor --var target="auth module" --var goal="Extract middleware"
```

---

## 🍳 Formulas

Formulas are TOML templates that generate related tasks with dependencies.

### Example: `bugfix.toml`

```toml
description = "Standard bug fix workflow"
formula = "bugfix"
version = 1

[vars.bug_description]
description = "Description of the bug"
required = true

[[steps]]
id = "reproduce"
title = "Reproduce the bug"
description = '''
{{bug_description}}

Write a failing test that reproduces this bug.'''

[[steps]]
id = "fix"
title = "Implement the fix"
description = "Fix the bug so the test passes."
needs = ["reproduce"]

[[steps]]
id = "verify"
title = "Verify fix"
description = "Run all tests, ensure no regressions."
needs = ["fix"]
```

### Built-in Formulas

| Formula | Steps | Use Case |
|---------|-------|----------|
| `bugfix` | reproduce → fix → verify | Fixing bugs |
| `feature` | design → implement → test → document | New features |
| `refactor` | analyze → test-before → refactor → verify | Code refactoring |

### Creating Custom Formulas

```bash
# Create a new formula
cat > workspace/.ralph/formulas/security-audit.toml << 'EOF'
description = "Security audit workflow"
formula = "security-audit"
version = 1

[[steps]]
id = "scan"
title = "Run security scanners"
description = "Run SAST tools and dependency audit"

[[steps]]
id = "review"
title = "Manual code review"
description = "Review high-risk areas"
needs = ["scan"]

[[steps]]
id = "fix"
title = "Fix vulnerabilities"
description = "Address identified issues"
needs = ["review"]
EOF
```

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  HOST                                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  openhands.py (v5.0.0) — TUI Manager                         │  │
│  │  ├── GitStateManager    ← State in git refs/tags/notes       │  │
│  │  ├── TaskManager        ← Bead-style IDs (oh-xxxxx)          │  │
│  │  ├── FormulaManager     ← TOML workflow templates            │  │
│  │  └── RalphManager       ← Controls daemon lifecycle          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│            │ docker exec                                            │
│            ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  DOCKER CONTAINER                                             │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  ralph_daemon.py (v3.0.0)                               │  │  │
│  │  │  ├── Git-native state functions                         │  │  │
│  │  │  ├── HierarchicalMemory (hot/warm/cold)                │  │  │
│  │  │  ├── SemanticSearch (sentence-transformers)            │  │  │
│  │  │  ├── ContextCondenser (LLM summarization)              │  │  │
│  │  │  └── StuckDetector (recovery strategies)               │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  │           │ spawns                                            │  │
│  │           ▼                                                   │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  OpenHands Agent Sessions (per iteration)              │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🔒 Security

v5.0 includes 100+ security fixes across multiple review rounds:

### Fixed Vulnerabilities
- ✅ Path traversal in git refs
- ✅ Shell injection in subprocess calls
- ✅ Command injection via heredoc
- ✅ Session ID injection
- ✅ MCP config newline injection
- ✅ PID file race conditions
- ✅ Unbounded file reads (OOM)
- ✅ ReDoS in regex patterns
- ✅ Non-atomic file writes

### Security Features
- Input sanitization for all git operations
- Base64 encoding for shell-unsafe content
- File locking for concurrent access
- Size limits on all file operations
- Symlink attack prevention

---

## ⚙️ Configuration

### LLM Configuration

Create `config/.openhands/agent_settings.json`:

```json
{
  "llm": {
    "model": "anthropic/claude-sonnet-4-20250514",
    "api_key": "your-api-key"
  },
  "agent": {
    "type": "CodeActAgent"
  }
}
```

### Environment Variables

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## 📊 Task Management

### Task Format (v2)

```json
{
  "version": 2,
  "project": "myapp",
  "tasks": {
    "oh-a1b2c": {
      "title": "Setup authentication",
      "description": "Implement JWT auth",
      "status": "done",
      "depends": []
    },
    "oh-d3e4f": {
      "title": "Add user profile",
      "status": "active",
      "depends": ["oh-a1b2c"]
    }
  }
}
```

### Task Statuses
- `pending` — Not started, waiting for dependencies
- `active` — Currently being worked on
- `done` — Completed successfully
- `failed` — Failed, needs attention
- `blocked` — Blocked by external factor

---

## 🛠️ Troubleshooting

### Check Ralph Status
```bash
# In container
cat /workspace/.ralph/heartbeat
cat .git/ralph/status
git log --oneline -5 --grep="[Ralph:Iter:"
```

### View Learnings
```bash
git log --show-notes=learnings -10
```

### Recovery from Crash
```bash
# Check latest checkpoint
git tag -l "ralph/cp/*" | tail -1
git show ralph/cp/iter-42
```

### Reset State
```bash
# Clear all Ralph state
rm -rf .git/ralph/
git notes --ref=learnings remove --all
```

---

## 📈 Version History

### v5.0.0 (Current)
- Git-native state management
- Bead-style task IDs
- Formula system (TOML templates)
- 100+ security fixes
- Removed file rotation (git handles history)

### v4.0.0
- Container-native daemon
- Hierarchical memory
- Semantic search
- Context condensing

### v3.0.0
- Initial Ralph daemon
- Basic task management

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Run tests (`python -m pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push branch (`git push origin feature/amazing`)
6. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- [OpenHands](https://github.com/All-Hands-AI/OpenHands) — AI coding agent
- [Gastown](https://github.com/steveyegge/gastown) — Inspiration for git-native state
- [Textual](https://github.com/Textualize/textual) — TUI framework

---

<div align="center">

**Built with ❤️ for autonomous coding**

[⬆ Back to Top](#-openhands-max)

</div>
