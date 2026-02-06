# 🤖 OpenHands Manager

<div align="center">

![Version](https://img.shields.io/badge/version-4.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

**A powerful TUI for managing OpenHands AI agent sessions with Ralph autonomous daemon**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Ralph Mode](#-ralph-autonomous-daemon) • [Architecture](#-architecture)

</div>

---

## ✨ Features

### 🖥️ Terminal User Interface
- **Project Management** — Create, configure, and manage multiple AI coding projects
- **Container Control** — Start, stop, restart, and shell into Docker containers
- **Session Management** — Background sessions that survive terminal close (tmux-based)
- **Real-time Monitoring** — Watch agent progress with live output updates

### 🤖 Ralph Autonomous Daemon
- **Container-Native** — Daemon runs inside Docker container, survives TUI restarts
- **Autonomous Coding** — AI agent works independently on complex tasks
- **Task Planning** — Automatically breaks down projects into manageable tasks
- **Architect Reviews** — Periodic code quality and architecture reviews
- **Self-Healing** — Stuck detection with automatic recovery strategies
- **Watchdog** — Cron-based watchdog ensures daemon stays alive

### 🧠 Smart Context Management
- **200K Token Support** — Optimized for large context models (Claude, GPT-4, etc.)
- **Hierarchical Memory** — Hot/warm/cold tiers for efficient context usage
- **Semantic Search** — Find relevant code using sentence-transformers
- **Knowledge Retention** — Learns from past iterations and mistakes
- **Context Condensing** — Automatic summarization with LLM verification

### 🔌 MCP Integration
- **Tool Servers** — Connect external tools via Model Context Protocol
- **Auto-Setup** — Automatic MCP gateway configuration
- **Skills System** — Extensible capabilities through skills

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Docker (running)
- 4GB+ RAM recommended

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/openhands-manager.git
cd openhands-manager

# Run (dependencies auto-install)
python3 openhands.py
```

That's it! The script automatically installs required dependencies:
- `textual` — TUI framework
- `sentence-transformers` — Semantic search (~500MB with PyTorch)

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

### Command Line Options
```bash
python3 openhands.py --help
python3 openhands.py --version
python3 openhands.py --list              # List all projects
python3 openhands.py myproject           # Quick-start project session
```

---

## 🎮 TUI Navigation

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `n` | New Project |
| `s` | Start Session |
| `r` | Start Ralph |
| `p` | Project Settings |
| `c` | Container Management |
| `F5` | Refresh |

### Main Screen
```
┌─────────────────────────────────────────────────────┐
│  Projects          │  Project Details              │
│  ──────────        │  ──────────────               │
│  > myproject       │  Name: myproject              │
│    webapp          │  Status: running              │
│    api-service     │  Container: oh-myproject      │
│                    │                               │
│  [+ New Project]   │  Workspace Files:             │
│  [> Start Session] │  ├── src/                     │
│  [R Start Ralph]   │  ├── tests/                   │
│  [* Settings]      │  └── README.md                │
│  [# Containers]    │                               │
└─────────────────────────────────────────────────────┘
```

---

## 🤖 Ralph Autonomous Daemon

Ralph is an autonomous AI coding daemon that runs inside the Docker container. It survives TUI restarts and works independently on complex projects.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Host Machine                                                    │
│  ┌───────────────────┐                                          │
│  │  OpenHands TUI    │  ←── Start/Stop/Monitor                  │
│  └─────────┬─────────┘                                          │
│            │                                                     │
│            ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Docker Container (openhands-runtime)                       ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Ralph Daemon (ralph_daemon.py)                      │   ││
│  │  │  ├── HierarchicalMemory (hot/warm/cold)             │   ││
│  │  │  ├── ContextCondenser (LLM summarization)           │   ││
│  │  │  ├── SemanticSearch (sentence-transformers)         │   ││
│  │  │  ├── StuckDetector (recovery strategies)            │   ││
│  │  │  └── CircuitBreaker (service resilience)            │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐  ││
│  │  │ Watchdog (cron) │  │  OpenHands Agent Sessions       │  ││
│  │  │ (auto-restart)  │  │  (spawned per iteration)        │  ││
│  │  └─────────────────┘  └─────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐
│  Planning   │ → │   Worker    │ → │  Architect  │ → │  Verify  │
│  Phase      │    │  Iterations │    │  Review     │    │  Phase   │
└─────────────┘    └─────────────┘    └─────────────┘    └──────────┘
       ↓                  ↓                  ↓                ↓
   Analyze code      Execute tasks      Check quality    Final tests
   Create PRD        Git commits        Fix issues       Mark done
```

1. **Planning Phase** — Analyzes codebase, creates PRD with task breakdown
2. **Worker Iterations** — Executes tasks one by one with git commits
3. **Architect Reviews** — Every N iterations, reviews code quality
4. **Context Condensing** — Periodically summarizes to stay within token limits
5. **Verification** — Final testing and validation

### Starting Ralph

1. Select a project in TUI
2. Press `r` or click "Start Ralph"
3. Configure:
   - **Task Description** — What should Ralph build?
   - **Max Iterations** — Limit iterations (0 = unlimited)
   - **Architect Interval** — Review frequency (default: 10)
   - **Condense Interval** — Context summarization frequency (default: 15)
4. Click "Start Ralph"

First run installs `sentence-transformers` (~500MB) — this takes 2-5 minutes.

### Ralph Configuration

```json
{
  "status": "running",
  "iteration": 15,
  "maxIterations": 0,
  "architectInterval": 10,
  "condenseInterval": 15
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `maxIterations` | 0 | Limit iterations (0 = unlimited) |
| `architectInterval` | 10 | Architect review every N iterations |
| `condenseInterval` | 15 | Context condensation frequency |

### Ralph Monitor

Press `r` while Ralph is running to open the monitor:

```
┌────────────────────────────────────────────────────────────────┐
│  Ralph: myproject                    Status: RUNNING           │
│  Iteration: 15 | Task: TASK-007      Heartbeat: 5s ago        │
├────────────────────────────────────────────────────────────────┤
│  Current Task                                                   │
│  ─────────────                                                  │
│  TASK-007: Implement user authentication                        │
│  - Add login/logout endpoints                                   │
│  - JWT token validation                                         │
├────────────────────────────────────────────────────────────────┤
│  Recent Output                                                  │
│  ─────────────                                                  │
│  [12:45:32] Creating auth middleware...                        │
│  [12:45:45] Added JWT verification                             │
│  [12:46:01] Committing changes                                 │
├────────────────────────────────────────────────────────────────┤
│  [P]ause  [S]top  [L]ogs  [R]efresh                   [Esc]   │
└────────────────────────────────────────────────────────────────┘
```

### Monitor Controls

| Key | Action |
|-----|--------|
| `p` | Pause/Resume Ralph |
| `s` | Stop Ralph daemon |
| `l` | View iteration logs |
| `r` | Refresh display |
| `Esc` | Go back (daemon keeps running!) |

### Daemon Lifecycle

- **Start**: TUI copies `ralph_daemon.py` to container, installs dependencies, starts daemon
- **Running**: Daemon runs independently, writes heartbeat every 30s
- **Watchdog**: Cron job checks daemon every minute, restarts if crashed
- **Stop**: TUI sends stop signal, daemon gracefully shuts down
- **Resume**: TUI can reconnect to running daemon after restart

---

## 📁 Project Structure

### Project Directory
```
~/openhands/projects/myproject/
├── workspace/                    # Your code lives here (mounted in container)
│   ├── src/
│   ├── .ralph/                   # Ralph state directory
│   │   ├── config.json           # Runtime config (daemon status, iteration)
│   │   ├── prd.json              # Task list (PRD)
│   │   ├── MISSION.md            # Project goal
│   │   ├── LEARNINGS.md          # Accumulated knowledge
│   │   ├── ARCHITECTURE.md       # Architecture documentation
│   │   ├── ralph_daemon.py       # Daemon script (copied from templates)
│   │   ├── ralph_daemon.log      # Daemon output log
│   │   ├── ralph_daemon.pid      # Daemon process ID
│   │   ├── heartbeat             # Last heartbeat timestamp
│   │   ├── iterations/           # Per-iteration logs
│   │   │   ├── iteration_001.log
│   │   │   └── ...
│   │   ├── memory/               # Hierarchical memory storage
│   │   │   ├── hot/              # Recent context
│   │   │   ├── warm/             # Important context
│   │   │   └── cold/             # Archived context
│   │   └── prompts/              # Prompt templates
│   └── ...
├── config/                       # Project config (mounted as /root)
│   └── .openhands/
│       ├── agent_settings.json
│       └── mcp_servers.json
└── data/                         # Persistent data
```

### Ralph Files

| File | Purpose |
|------|---------|
| `config.json` | Runtime state: status, iteration, settings |
| `prd.json` | PRD with task list, dependencies, status |
| `MISSION.md` | Original project goal/description |
| `LEARNINGS.md` | Knowledge accumulated during development |
| `ARCHITECTURE.md` | Architecture decisions and patterns |
| `ralph_daemon.py` | The daemon script (auto-copied) |
| `ralph_daemon.log` | Daemon stdout/stderr output |
| `heartbeat` | Unix timestamp of last heartbeat |
| `iterations/` | Per-iteration detailed logs |
| `memory/` | Hierarchical context storage |

---

## ⚙️ Configuration

### LLM Configuration

Create `config/.openhands/agent_settings.json`:

```json
{
  "llm": {
    "model": "anthropic/claude-sonnet-4-20250514",
    "api_key": "your-api-key",
    "base_url": null
  },
  "agent": {
    "type": "CodeActAgent"
  }
}
```

### MCP Servers

Create `config/.openhands/mcp_servers.json`:

```json
{
  "servers": {
    "memory": {
      "transport": "uvx",
      "command": "mcp-server-memory"
    },
    "filesystem": {
      "transport": "uvx",
      "command": "mcp-server-filesystem",
      "args": ["--root", "/workspace"]
    }
  }
}
```

### Templates

The manager includes templates for common configurations:

```
~/openhands/templates/
├── llm/
│   ├── anthropic.json
│   ├── openai.json
│   └── local.json
├── mcp/
│   ├── basic.json
│   └── full.json
└── skills/
    └── ...
```

---

## 🐳 Container Management

### Container Screen

Press `c` to open container management:

```
┌────────────────────────────────────────────────────────────┐
│  Containers                                                │
│  ──────────                                                │
│  NAME             STATUS      IMAGE                        │
│  oh-myproject     running     openhands/runtime:latest     │
│  oh-webapp        stopped     openhands/runtime:latest     │
├────────────────────────────────────────────────────────────┤
│ [S]tart  [T]op  [R]estart  [D]elete  [H]Shell  [B]Back    │
└────────────────────────────────────────────────────────────┘
```

### Container Commands

| Key | Action |
|-----|--------|
| `s` | Start selected container |
| `t` | Stop selected container |
| `r` | Restart container |
| `d` | Delete container |
| `h` | Open shell in container |
| `b` | Go back |

---

## 🛠️ Advanced Usage

### Manual Daemon Control

```bash
# Start daemon manually in container
docker exec openhands-myproject bash -c "
  cd /workspace
  setsid python3 /workspace/.ralph/ralph_daemon.py >> /workspace/.ralph/ralph_daemon.log 2>&1 &
"

# Stop daemon
docker exec openhands-myproject pkill -f ralph_daemon.py

# Restart daemon
docker exec openhands-myproject bash -c "
  pkill -f ralph_daemon.py
  sleep 2
  setsid python3 /workspace/.ralph/ralph_daemon.py >> /workspace/.ralph/ralph_daemon.log 2>&1 &
"
```

### Watchdog Configuration

The watchdog runs via cron inside the container:
```bash
# View watchdog cron
docker exec openhands-myproject crontab -l

# Disable watchdog
docker exec openhands-myproject bash -c "crontab -l | grep -v ralph_watchdog | crontab -"

# Re-enable watchdog
docker exec openhands-myproject bash -c "
  (crontab -l 2>/dev/null; echo '* * * * * /workspace/.ralph/ralph_watchdog.sh >> /workspace/.ralph/watchdog.log 2>&1') | crontab -
"
```

### Edit PRD Manually

```bash
# Edit task list
nano ~/openhands/projects/myproject/workspace/.ralph/prd.json

# Mark task as done
# Change "passes": false to "passes": true
```

### Resume After TUI Restart

The daemon keeps running even if you close the TUI. Just reopen and connect:
```bash
python3 openhands.py
# Select project → Press 'r' → Monitor shows running daemon
```

---

## 🔧 Troubleshooting

### Common Issues

#### "Docker not available"
```bash
# Start Docker daemon
sudo systemctl start docker

# Or on macOS
open -a Docker
```

#### "Failed to start container"
```bash
# Check Docker status
docker info

# Pull latest image
docker pull docker.openhands.dev/openhands/runtime:latest-nikolaik
```

#### "No space left on device"
```bash
# sentence-transformers requires ~1GB for installation
# Free up Docker space:
docker system prune -a

# Check disk space
df -h
```

#### "Daemon failed to start"
```bash
# Check daemon log in container
docker exec openhands-myproject cat /workspace/.ralph/ralph_daemon.log

# Check if dependencies installed
docker exec openhands-myproject python3 -c "import sentence_transformers; print('OK')"

# Manually start daemon for debugging
docker exec -it openhands-myproject python3 /workspace/.ralph/ralph_daemon.py
```

#### "Daemon keeps crashing"
```bash
# Check watchdog log
docker exec openhands-myproject cat /workspace/.ralph/watchdog.log

# Check system resources in container
docker exec openhands-myproject free -h
docker exec openhands-myproject df -h
```

#### Ralph stuck on same task
1. Check iteration logs: `workspace/.ralph/iterations/`
2. Review `LEARNINGS.md` for error patterns
3. Stop daemon, edit PRD to skip task, restart

### Logs

```bash
# Daemon log (main output)
cat ~/openhands/projects/myproject/workspace/.ralph/ralph_daemon.log

# Per-iteration logs
ls ~/openhands/projects/myproject/workspace/.ralph/iterations/

# Watchdog log
cat ~/openhands/projects/myproject/workspace/.ralph/watchdog.log

# Container logs
docker logs openhands-myproject
```

### Debug Commands

```bash
# Check daemon status
docker exec openhands-myproject pgrep -f ralph_daemon.py

# Check heartbeat age
docker exec openhands-myproject cat /workspace/.ralph/heartbeat

# Check config
docker exec openhands-myproject cat /workspace/.ralph/config.json

# Interactive shell
docker exec -it openhands-myproject bash
```

---

## 📊 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  HOST MACHINE                                                         │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  OpenHands Manager (openhands.py)                              │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐ │  │
│  │  │ TUI (Textual)│ │ Docker API   │ │ Project Manager        │ │  │
│  │  │ - Screens    │ │ - exec       │ │ - Create/configure     │ │  │
│  │  │ - Monitors   │ │ - cp         │ │ - Templates            │ │  │
│  │  │ - Dialogs    │ │ - start/stop │ │ - Settings             │ │  │
│  │  └──────────────┘ └──────────────┘ └────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────┘  │
│         │                    │                                        │
│         │   docker exec      │   bind mount                          │
│         ▼                    ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  DOCKER CONTAINER (openhands-runtime)                          │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Ralph Daemon (/workspace/.ralph/ralph_daemon.py)       │  │  │
│  │  │                                                          │  │  │
│  │  │  Components:                                             │  │  │
│  │  │  ├── RalphDaemon          Main loop, iteration control  │  │  │
│  │  │  ├── HierarchicalMemory   Hot/warm/cold context tiers   │  │  │
│  │  │  ├── ContextCondenser     LLM-powered summarization     │  │  │
│  │  │  ├── SemanticSearch       sentence-transformers         │  │  │
│  │  │  ├── LearningsManager     Knowledge accumulation        │  │  │
│  │  │  ├── StuckDetector        Recovery strategies           │  │  │
│  │  │  └── CircuitBreaker       Service resilience            │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │         │                                                      │  │
│  │         │ spawns                                               │  │
│  │         ▼                                                      │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  OpenHands Agent Sessions                               │  │  │
│  │  │  (created per iteration via openhands CLI)              │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │  │
│  │  │ Watchdog     │  │ MCP Gateway  │  │ /workspace       │    │  │
│  │  │ (cron)       │  │ (tools)      │  │ (your code)      │    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **TUI → Container**: `docker exec` to start/stop daemon, read status
2. **Daemon → OpenHands**: Spawns agent sessions via `openhands` CLI
3. **Daemon → Files**: Writes state to `/workspace/.ralph/` (visible on host)
4. **Watchdog → Daemon**: Cron checks heartbeat, restarts if stale

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [OpenHands](https://github.com/All-Hands-AI/OpenHands) — The AI coding agent
- [Textual](https://github.com/Textualize/textual) — TUI framework
- [sentence-transformers](https://www.sbert.net/) — Semantic search

---

<div align="center">


[⬆ Back to Top](#-openhands-manager)

</div>
