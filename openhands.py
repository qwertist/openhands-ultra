#!/usr/bin/env python3
"""
OpenHands Manager - TUI for OpenHands with Ralph autonomous loop

Features:
- Project management (create, delete, list)
- Container management (start, stop, shell, logs)
- Ralph v3 autonomous coding loop with large context optimization
- LLM/MCP/Skills templates
- Background sessions (tmux-based, survives terminal close)
- Graceful shutdown and state preservation

Usage:
    python3 openhands.py              # Launch TUI
    python3 openhands.py <project>    # Quick start project session
    python3 openhands.py --help       # Show help
"""

import base64
import fcntl
import hashlib
import json
import logging
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid  # Added for UUID-based temp file names
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import deque  # Added for LRU cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('openhands')

# =============================================================================
# AUTO-INSTALL DEPENDENCIES
# =============================================================================

def install_dependencies():
    """Install required packages if not present."""
    # Map of import name -> pip package name
    required = {
        'textual': 'textual',
        'sentence_transformers': 'sentence-transformers',
    }
    missing = []
    
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        print(f"Installing dependencies: {', '.join(missing)}...")
        print("(sentence-transformers is ~500MB with torch, please wait...)")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing, "--break-system-packages"
            ])
            print("Done! Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install dependencies: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Unexpected error during dependency installation: {e}")
            sys.exit(1)

# Now import textual
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Button, DataTable, DirectoryTree, Footer, Header, Input, Label,
    ListItem, ListView, Log, ProgressBar, RichLog, Select, Static, 
    Switch, TextArea, Tree
)
from textual import work
from textual.reactive import reactive

# Only install dependencies when running as main module
# This avoids circular import issues during module loading and importlib introspection
if __name__ == "__main__":
    install_dependencies()

# =============================================================================
# CONFIGURATION
# =============================================================================

VERSION = "4.0.0"  # Container-native daemon architecture

# Runtime image - contains Python, Node, tools for agent
RUNTIME_IMAGE = "docker.openhands.dev/openhands/runtime:latest-nikolaik"
DOCKER_TIMEOUT = 28800  # 8 hours max session timeout

# PATH for tools installed via uv/pip inside container
CONTAINER_PATH = "/root/.local/bin:/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"

# =============================================================================
# CONTEXT LIMITS (optimized for 250K+ token models)
# =============================================================================
# Budget allocation for 250K context:
#   - Ralph prompt content: ~100-150K tokens (main knowledge)
#   - OpenHands system/tools: ~50K tokens  
#   - Model working memory: ~50K tokens (for reasoning)
#   - Safety buffer: ~20K tokens
# =============================================================================

MAX_PROMPT_TOKENS = 125000  # Our budget for 250K models
CHARS_PER_TOKEN = 4
MAX_PROMPT_CHARS = MAX_PROMPT_TOKENS * CHARS_PER_TOKEN

# Knowledge retention limits (chars) - optimized for 250K context
# More context = smarter decisions, but avoid attention dilution
CRITICAL_LEARNINGS_LIMIT = 80000   # ~20K tokens - errors, patterns (doubled)
RECENT_LEARNINGS_LIMIT = 80000     # ~20K tokens - recent discoveries  
ARCHITECTURE_LIMIT = 60000         # ~15K tokens - project structure
GUARDRAILS_LIMIT = 30000           # ~7.5K tokens - mistakes to avoid
MEMORY_CONTEXT_LIMIT = 60000       # ~15K tokens - iteration history
MISSION_LIMIT = 50000              # ~12.5K tokens - full mission context

# Cleanup thresholds
MAX_ITERATION_FILES = 1000
MAX_LOG_SIZE_BYTES = 500000
MAX_PROGRESS_LINES = 1000

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 5.0
MAX_DELAY = 60.0

# Disk space monitoring
MIN_FREE_SPACE_MB = 500

# Project name validation (safe for systemd, cron, filesystem)
VALID_PROJECT_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')

def validate_project_name(name: str) -> Tuple[bool, str]:
    """Validate project name for safety in systemd, cron, filesystem."""
    if not name:
        return False, "Project name cannot be empty"
    if len(name) > 64:
        return False, "Project name too long (max 64 characters)"
    if not VALID_PROJECT_NAME_PATTERN.match(name):
        return False, "Project name can only contain letters, numbers, underscores, and hyphens (must start with letter/number)"
    return True, ""

# =============================================================================
# OPENHANDS INSTALL SCRIPT
# =============================================================================

OPENHANDS_INSTALL_SCRIPT = '''
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# Install required system packages (jq, lsof for MCP, nano for editing, nodejs for supergateway)
echo "Installing system dependencies..."

# Update package lists first (CRITICAL!)
echo "Updating package lists..."
apt-get update -qq 2>/dev/null || {
    echo "ERROR: apt-get update failed!"
    echo "Trying with non-interactive frontend..."
    DEBIAN_FRONTEND=noninteractive apt-get update -qq 2>/dev/null || {
        echo "ERROR: apt-get update failed completely. Cannot install packages."
        exit 1
    }
}
echo "Package lists updated."

# Critical packages for MCP - must install successfully
echo "Installing jq (required for MCP)..."
apt-get install -y -qq jq curl 2>/dev/null || {
    echo "ERROR: Failed to install jq (required for MCP)"
    exit 1
}

# Optional packages (nice to have but not critical)
apt-get install -y -qq nano lsof 2>/dev/null || true

# Install nodejs/npm for supergateway
if ! command -v npm &>/dev/null; then
    echo "Installing nodejs/npm for supergateway..."
    apt-get install -y -qq nodejs npm 2>/dev/null || {
        echo "WARNING: Failed to install nodejs/npm - supergateway may not work"
    }
fi

# Install supergateway for MCP
if command -v npm &>/dev/null && ! command -v supergateway &>/dev/null; then
    echo "Installing supergateway..."
    npm install -g supergateway 2>&1 || {
        echo "WARNING: Failed to install supergateway"
    }
fi

# Show installed versions
echo ""
echo "=== Installed versions ==="
echo "jq: $(jq --version 2>/dev/null || echo 'not found')"
echo "curl: $(curl --version 2>/dev/null | head -1 || echo 'not found')"
echo "npm: $(npm --version 2>/dev/null || echo 'not found')"
echo "supergateway: $(supergateway --version 2>/dev/null || echo 'not found')"
echo "=========================="
echo ""

if ! command -v openhands &> /dev/null; then
    echo "Installing OpenHands..."
    pip install -q uv 2>/dev/null || true
    timeout 300 uv tool install openhands --python 3.12 2>&1 | tail -3
    uv tool update-shell 2>/dev/null || true
    export PATH="/root/.local/bin:$PATH"
    
    if ! command -v openhands &> /dev/null && [ ! -x "/root/.local/bin/openhands" ]; then
        echo "ERROR: OpenHands installation failed"
        exit 1
    fi
    echo "OpenHands installed!"
fi
'''

# MCP Gateway setup script - FAST VERSION (assumes jq/supergateway already installed)
MCP_GATEWAY_SCRIPT = r'''
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
MCP_SERVERS="/root/.openhands/mcp_servers.json"
GATEWAY_PIDS="/root/.openhands/.gateway_pids"
GATEWAY_STATE="/root/.openhands/.gateway_state"
MCP_JSON="/root/.openhands/mcp.json"

echo "=== MCP Gateway Setup ==="
echo "MCP_SERVERS: $MCP_SERVERS"

# Quick check - should already be installed by setup
if ! command -v jq &>/dev/null; then
    echo "ERROR: jq not found! Run setup first."
    exit 1
fi
if ! command -v supergateway &>/dev/null; then
    echo "ERROR: supergateway not found! Run setup first."
    exit 1
fi
echo "Prerequisites: OK"

if [ ! -f "$MCP_SERVERS" ]; then
    echo "No mcp_servers.json found, skipping gateway setup"
    exit 0
fi

echo "Found mcp_servers.json, starting gateways..."

# Kill any existing gateways (quick)
echo "Stopping existing gateways..."
killall -9 supergateway 2>/dev/null || true
killall -9 mcp-server 2>/dev/null || true
killall -9 mcp-memory 2>/dev/null || true
killall -9 memory-server 2>/dev/null || true
rm -f "$GATEWAY_PIDS"
echo "Gateways stopped."

# Read servers from config
echo "Reading MCP config..."
servers=$(jq -r '.mcpServers | keys[]' "$MCP_SERVERS" 2>/dev/null)
if [ -z "$servers" ]; then
    echo "No servers found in mcp_servers.json"
    exit 0
fi

echo "Servers to start: $servers"

# Build SSE servers array
sse_servers="[]"
port=8001

for server in $servers; do
    cmd=$(jq -r --arg s "$server" '.mcpServers[$s].command // empty' "$MCP_SERVERS")
    args=$(jq -r --arg s "$server" '.mcpServers[$s].args // [] | .[]' "$MCP_SERVERS" | tr '\n' ' ')
    
    if [ -z "$cmd" ]; then
        echo "Skipping $server (no command)"
        continue
    fi
    
    echo "Starting $server on port $port: $cmd $args"
    
    full_cmd="$cmd $args"
    nohup supergateway --stdio "$full_cmd" --port $port > /tmp/gateway_$server.log 2>&1 &
    echo $! >> "$GATEWAY_PIDS"
    
    sse_servers=$(echo "$sse_servers" | jq --arg url "http://localhost:$port/sse" '. + [$url]')
    port=$((port + 1))
done

# Create mcp.json for OpenHands
echo "Creating mcp.json..."
jq -n --argjson sse "$sse_servers" '{sse_servers: $sse}' > "$MCP_JSON"
cp "$MCP_JSON" "$GATEWAY_STATE"

echo "mcp.json created:"
cat "$MCP_JSON"

# HIGH PRIORITY FIX: Health check polling instead of fixed sleep
# This handles slow services like memory-service that needs to download ONNX models
echo "Waiting for gateways to start (max 60s)..."
max_wait=60
for ((i=0; i<max_wait; i++)); do
    sleep 1
    # Check if at least one gateway is responding
    if [ -f "$GATEWAY_PIDS" ]; then
        all_ready=true
        port=8001
        for server in $servers; do
            if curl -sf "http://localhost:$port/sse" > /dev/null 2>&1 || kill -0 $(sed -n "${port##??}p" "$GATEWAY_PIDS" 2>/dev/null) 2>/dev/null; then
                : # gateway responding or process exists
            else
                all_ready=false
                break
            fi
            port=$((port + 1))
        done
        if $all_ready; then
            echo "All gateways ready after ${i}s"
            break
        fi
    fi
done

# Check status with more detail
if [ -f "$GATEWAY_PIDS" ]; then
    running=0
    total=0
    server_idx=0
    server_names=$(jq -r '.mcpServers | keys[]' "$MCP_SERVERS" 2>/dev/null)
    
    while read pid; do
        total=$((total + 1))
        server_name=$(echo "$server_names" | sed -n "${total}p")
        if kill -0 $pid 2>/dev/null; then
            running=$((running + 1))
            echo "  ✓ $server_name (PID $pid) - running"
        else
            echo "  ✗ $server_name (PID $pid) - DEAD"
            # Show last lines of log if exists
            log_file="/tmp/gateway_$server_name.log"
            if [ -f "$log_file" ]; then
                echo "    Log tail:"
                tail -5 "$log_file" | sed 's/^/      /'
            fi
        fi
    done < "$GATEWAY_PIDS"
    echo "Gateways: $running/$total running"
else
    echo "No gateway PIDs file found - no gateways started"
fi

echo "=== MCP Gateway Setup Complete ==="
'''

# MCP warmup script
MCP_WARMUP_SCRIPT = '''
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
MCP_SERVERS="/root/.openhands/mcp_servers.json"
UV_TOOLS="/root/.local/share/uv/tools"

if [ ! -f "$MCP_SERVERS" ]; then
    echo "  No mcp_servers.json found, skipping warmup"
    exit 0
fi

# Check if jq is available
if ! command -v jq &>/dev/null; then
    echo "  jq not found, installing..."
    apt-get update -qq && apt-get install -y -qq jq 2>/dev/null || true
fi

if ! command -v jq &>/dev/null; then
    echo "  WARNING: jq installation failed, MCP servers may not work properly"
    exit 0
fi

echo "Setting up MCP servers..."

# Get uvx packages from config
uvx_pkgs=$(jq -r '.mcpServers // {} | to_entries[] | select(.value.command == "uvx") | .value.args | map(select(startswith("-") | not)) | .[0] // empty' "$MCP_SERVERS" 2>/dev/null | sort -u)
if [ -z "$uvx_pkgs" ]; then
    uvx_pkgs=$(jq -r '.stdio_servers // [] | .[] | select(.command == "uvx") | .args | map(select(startswith("-") | not)) | .[0] // empty' "$MCP_SERVERS" 2>/dev/null | sort -u)
fi

echo "  Found uvx packages: $uvx_pkgs"

# Install all uvx tools with timeout
echo ""
echo "  Installing MCP tools..."
for pkg in $uvx_pkgs; do
    if [ -n "$pkg" ]; then
        echo "    -> $pkg"
        timeout 120 uv tool install "$pkg" 2>&1 | tail -3 || echo "    (timeout or error, continuing...)"
    fi
done

# Install onnxruntime for memory service if present
for pkg in $uvx_pkgs; do
    if [ -n "$pkg" ]; then
        tool_pip="$UV_TOOLS/$pkg/bin/pip"
        if [ -f "$tool_pip" ]; then
            case "$pkg" in
                mcp-memory-service|*memory*)
                    echo "    -> $pkg: installing onnxruntime..."
                    timeout 60 "$tool_pip" install -q onnxruntime 2>&1 | tail -2 || true
                    ;;
            esac
        fi
    fi
done

# Get npx packages  
npx_pkgs=$(jq -r '.mcpServers // {} | to_entries[] | select(.value.command == "npx") | .value.args | map(select(startswith("-") | not)) | .[0] // empty' "$MCP_SERVERS" 2>/dev/null | sort -u)
if [ -z "$npx_pkgs" ]; then
    npx_pkgs=$(jq -r '.stdio_servers // [] | .[] | select(.command == "npx") | .args | map(select(startswith("-") | not)) | .[0] // empty' "$MCP_SERVERS" 2>/dev/null | sort -u)
fi

echo "  Found npx packages: $npx_pkgs"

for pkg in $npx_pkgs; do
    if [ -n "$pkg" ]; then
        echo "    -> Installing $pkg..."
        timeout 180 npm install -g "$pkg" 2>&1 | tail -2 || echo "    (timeout or error, continuing...)"
    fi
done

echo ""
echo "  MCP tools installed! Runtime deps will download when gateways start."
'''

# Tmux session names
TMUX_OH_SESSION = "openhands"


# Get script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECTS_DIR = SCRIPT_DIR / "projects"
TEMPLATES_DIR = SCRIPT_DIR / "templates"
LLM_TEMPLATES_DIR = TEMPLATES_DIR / "llm"
MCP_TEMPLATES_DIR = TEMPLATES_DIR / "mcp"
RALPH_PROMPTS_DIR = TEMPLATES_DIR / "tools" / "ralph"
SKILLS_TEMPLATES_DIR = TEMPLATES_DIR / "skills"
LOG_FILE = SCRIPT_DIR / ".openhands.log"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Project:
    """Project data model."""
    name: str
    path: Path
    
    @property
    def workspace(self) -> Path:
        return self.path / "workspace"
    
    @property
    def config_dir(self) -> Path:
        return self.path / "config"
    
    @property
    def openhands_dir(self) -> Path:
        return self.config_dir / ".openhands"
    
    @property
    def ralph_dir(self) -> Path:
        return self.workspace / ".ralph"
    
    @property
    def container_name(self) -> str:
        return f"openhands-{self.name}"
    
    @property
    def exists(self) -> bool:
        return self.path.exists() and self.workspace.exists()
    
    @property
    def has_ralph(self) -> bool:
        return self.ralph_dir.exists()
    
    @property
    def is_persistent(self) -> bool:
        return (self.path / ".persistent").exists()
    
    def get_llm_model(self) -> str:
        """Get LLM model from agent_settings.json."""
        settings_file = self.openhands_dir / "agent_settings.json"
        if settings_file.exists():
            try:
                data = json.loads(settings_file.read_text())
                return data.get("LLM_MODEL", data.get("llm", {}).get("model", "not set"))
            except Exception:
                pass
        return "not configured"
    
    def get_container_status(self) -> str:
        """Get container status: running, stopped, none."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name=^{self.container_name}$", 
                 "--format", "{{.Status}}"],
                capture_output=True, text=True, timeout=5
            )
            status = result.stdout.strip()
            if not status:
                return "none"
            return "running" if status.startswith("Up") else "stopped"
        except Exception:
            return "unknown"
    
    def get_ralph_config(self) -> dict:
        """Get Ralph config from config.json."""
        config_file = self.ralph_dir / "config.json"
        if config_file.exists():
            try:
                return json.loads(config_file.read_text())
            except Exception:
                pass
        return {}
    
    def get_ralph_progress(self) -> Tuple[int, int]:
        """Get (done_tasks, total_tasks)."""
        # PRD is now in workspace/.ralph/
        prd_file = self.workspace / ".ralph" / "prd.json"
        if not prd_file.exists():
            # Fallback to old location for backwards compatibility
            prd_file = self.ralph_dir / "prd.json"
        
        if prd_file.exists():
            try:
                data = json.loads(prd_file.read_text())
                stories = data.get("userStories", [])
                total = len(stories)
                done = sum(1 for s in stories if s.get("passes", False))
                return (done, total)
            except Exception:
                pass
        return (0, 0)


@dataclass 
class RalphConfig:
    """Ralph configuration."""
    task_description: str = ""
    max_iterations: int = 0  # 0 = infinite
    session_timeout: int = 1800  # 30 minutes per session
    architect_interval: int = 10  # Review every N iterations
    condense_interval: int = 15  # Summarize context every N iterations (0 = disabled)
    condense_before_architect: bool = True  # Always condense before architect review
    require_verification: bool = True
    pause_between: int = 10  # Seconds between iterations


@dataclass
class Checkpoint:
    """Checkpoint for resuming Ralph loop."""
    iteration: int
    task_id: str
    prd_state: dict
    timestamp: str
    status: str  # "in_progress", "completed", "failed"


@dataclass
class RalphState:
    """Complete Ralph state for save/restore."""
    config: dict
    prd: dict
    checkpoint: Optional[Checkpoint]
    pending_tasks: List[str]  # Tasks queued while running
    timestamp: str


# =============================================================================
# STATE MANAGER - Handles save/restore of complete Ralph state
# =============================================================================

class StateManager:
    """Manages complete Ralph state for pause/resume and edge cases."""
    
    def __init__(self, ralph_dir: Path, workspace_ralph: Path = None):
        self.ralph_dir = ralph_dir
        self.workspace_ralph = workspace_ralph  # Will be set by RalphManager
        self.state_file = ralph_dir / "state.json"
        self.backup_dir = ralph_dir / "backups"
        # Directory is created by RalphManager via Docker
        self.pending_tasks_file = ralph_dir / "pending_tasks.json"
    
    def set_workspace(self, workspace_ralph: Path):
        """Set workspace .ralph directory for git-tracked files."""
        self.workspace_ralph = workspace_ralph
    
    def save_state(self, ralph_manager: 'RalphManager') -> bool:
        """Save complete Ralph state for later restore."""
        try:
            state = RalphState(
                config=ralph_manager.get_config(),
                prd=ralph_manager.get_prd(),
                checkpoint=ralph_manager.load_checkpoint(),
                pending_tasks=self.get_pending_tasks(),
                timestamp=datetime.now().isoformat()
            )
            
            # Save to state file
            safe_write_json(self.state_file, {
                "config": state.config,
                "prd": state.prd,
                "checkpoint": state.checkpoint.__dict__ if state.checkpoint else None,
                "pending_tasks": state.pending_tasks,
                "timestamp": state.timestamp
            })
            
            # Also save as backup with timestamp
            backup_file = self.backup_dir / f"state_{int(time.time())}.json"
            shutil.copy(self.state_file, backup_file)
            
            # Keep only last 10 backups
            backups = sorted(self.backup_dir.glob("state_*.json"))
            for old_backup in backups[:-10]:
                old_backup.unlink()
            
            return True
        except Exception as e:
            log_error(f"Failed to save state: {e}")
            return False
    
    def load_state(self) -> Optional[RalphState]:
        """Load saved Ralph state."""
        if not self.state_file.exists():
            return None
        
        try:
            data = json.loads(self.state_file.read_text())
            checkpoint_data = data.get("checkpoint")
            checkpoint = None
            if checkpoint_data:
                checkpoint = Checkpoint(
                    iteration=checkpoint_data.get("iteration", 0),
                    task_id=checkpoint_data.get("task_id", ""),
                    prd_state=checkpoint_data.get("prd_state", {}),
                    timestamp=checkpoint_data.get("timestamp", ""),
                    status=checkpoint_data.get("status", "unknown")
                )
            
            return RalphState(
                config=data.get("config", {}),
                prd=data.get("prd", {}),
                checkpoint=checkpoint,
                pending_tasks=data.get("pending_tasks", []),
                timestamp=data.get("timestamp", "")
            )
        except Exception as e:
            log_error(f"Failed to load state: {e}")
            return None
    
    def restore_state(self, ralph_manager: 'RalphManager') -> bool:
        """Restore Ralph from saved state."""
        state = self.load_state()
        if not state:
            return False
        
        try:
            # Restore config
            safe_write_json(ralph_manager.config_file, state.config)
            
            # Restore PRD
            safe_write_json(ralph_manager.prd_file, state.prd)
            
            # Restore checkpoint
            if state.checkpoint:
                ralph_manager.save_checkpoint(
                    state.checkpoint.iteration,
                    state.checkpoint.task_id,
                    state.checkpoint.status
                )
            
            # Restore pending tasks
            self.set_pending_tasks(state.pending_tasks)
            
            return True
        except Exception as e:
            log_error(f"Failed to restore state: {e}")
            return False
    
    def get_pending_tasks(self) -> List[str]:
        """Get tasks queued while Ralph was running."""
        if not self.pending_tasks_file.exists():
            return []
        try:
            return json.loads(self.pending_tasks_file.read_text())
        except Exception:
            return []
    
    def add_pending_task(self, task: str) -> bool:
        """Add a task to the pending queue."""
        try:
            tasks = self.get_pending_tasks()
            tasks.append(task)
            safe_write_json(self.pending_tasks_file, tasks)
            return True
        except Exception as e:
            log_error(f"Failed to add pending task: {e}")
            return False
    
    def set_pending_tasks(self, tasks: List[str]) -> bool:
        """Set pending tasks list."""
        try:
            safe_write_json(self.pending_tasks_file, tasks)
            return True
        except Exception as e:
            log_error(f"Failed to set pending tasks: {e}")
            return False
    
    def clear_pending_tasks(self) -> bool:
        """Clear all pending tasks."""
        try:
            if self.pending_tasks_file.exists():
                self.pending_tasks_file.unlink()
            return True
        except Exception as e:
            log_error(f"Failed to clear pending tasks: {e}")
            return False
    
    def backup_prd(self) -> Path:
        """Create backup of prd.json from workspace."""
        # PRD is now in workspace/.ralph/
        prd_file = self.workspace_ralph / "prd.json" if self.workspace_ralph else self.ralph_dir / "prd.json"
        if not prd_file.exists():
            return None
        
        backup_file = self.backup_dir / f"prd_{int(time.time())}.json"
        shutil.copy(prd_file, backup_file)
        
        # Keep only last 20 PRD backups
        backups = sorted(self.backup_dir.glob("prd_*.json"))
        for old_backup in backups[:-20]:
            old_backup.unlink()
        
        return backup_file
    
    def restore_prd_from_backup(self, backup_file: Optional[Path] = None) -> bool:
        """Restore prd.json to workspace from backup."""
        # PRD is now in workspace/.ralph/
        prd_file = self.workspace_ralph / "prd.json" if self.workspace_ralph else self.ralph_dir / "prd.json"
        
        if backup_file and backup_file.exists():
            shutil.copy(backup_file, prd_file)
            return True
        
        # Find most recent backup
        backups = sorted(self.backup_dir.glob("prd_*.json"), reverse=True)
        if backups:
            shutil.copy(backups[0], prd_file)
            return True
        
        return False


# =============================================================================
# RESILIENCE UTILITIES - Atomic operations, health checks, circuit breakers
# =============================================================================

def atomic_write(filepath: Path, content: str, backup: bool = True) -> bool:
    """
    Write file atomically to prevent corruption on crash.
    
    Pattern: write to .tmp file, then rename (atomic on POSIX).
    Optionally keeps one backup.
    """
    filepath = Path(filepath)
    tmp_path = filepath.with_suffix(filepath.suffix + '.tmp')
    backup_path = filepath.with_suffix(filepath.suffix + '.bak') if backup else None
    
    try:
        # Write to temp file
        tmp_path.write_text(content)
        
        # Sync to disk
        with open(tmp_path, 'r', encoding='utf-8') as f:
            os.fsync(f.fileno())
        
        # Backup existing file if requested
        if backup and filepath.exists():
            try:
                if backup_path.exists():
                    backup_path.unlink()
                filepath.rename(backup_path)
            except Exception:
                pass  # Best effort backup
        
        # Atomic rename
        tmp_path.rename(filepath)
        return True
        
    except Exception as e:
        log_error(f"Atomic write failed for {filepath}: {e}")
        # Cleanup temp file
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


def safe_write_text(filepath: Path, content: str, backup: bool = False) -> bool:
    """
    Safely write text file.
    
    Uses atomic write for safety. If permission denied (file owned by root 
    from Docker), logs warning and returns False - caller should handle gracefully.
    
    Note: To avoid permission issues, host should create all workspace/.ralph/ 
    files BEFORE starting daemon. Daemon preserves ownership when updating.
    """
    filepath = Path(filepath)
    
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        return atomic_write(filepath, content, backup=backup)
    except PermissionError:
        # File owned by root (created in Docker) - can't write without sudo
        # Log warning but don't crash - this is expected if host didn't pre-create files
        log_error(f"Permission denied writing {filepath} (owned by root from Docker)")
        return False
    except Exception as e:
        log_error(f"Failed to write {filepath}: {e}")
        return False


def safe_append_text(filepath: Path, content: str) -> bool:
    """Safely append text to file. Returns False on permission error."""
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'a') as f:
            f.write(content)
        return True
    except PermissionError:
        log_error(f"Permission denied appending to {filepath}")
        return False
    except Exception as e:
        log_error(f"Failed to append to {filepath}: {e}")
        return False


def safe_read_json(filepath: Path, default: Any = None) -> Any:
    """
    Safely read JSON file with corruption recovery.
    
    Falls back to .bak file if main file is corrupted.
    """
    filepath = Path(filepath)
    backup_path = filepath.with_suffix(filepath.suffix + '.bak')
    
    # Try main file
    if filepath.exists():
        try:
            content = filepath.read_text()
            return json.loads(content)
        except json.JSONDecodeError as e:
            log_error(f"JSON corruption in {filepath}: {e}")
            # Try backup
            if backup_path.exists():
                try:
                    content = backup_path.read_text()
                    data = json.loads(content)
                    log(f"Recovered {filepath} from backup")
                    # Restore from backup
                    atomic_write(filepath, content, backup=False)
                    return data
                except Exception:
                    pass
    
    return default


def safe_write_json(filepath: Path, data: Any, indent: int = 2) -> bool:
    """Write JSON file atomically with backup and permission handling."""
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        return safe_write_text(filepath, content, backup=True)
    except Exception as e:
        log_error(f"Failed to write JSON {filepath}: {e}")
        return False


class CircuitBreaker:
    """
    Circuit breaker pattern for external services (API calls, Docker, etc).
    
    States:
    - CLOSED: normal operation
    - OPEN: failing, reject requests
    - HALF_OPEN: testing if recovered
    """
    
    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_timeout: float = 60.0, half_open_requests: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.failures = 0
        self.successes_in_half_open = 0
        self.state = "CLOSED"
        self.last_failure_time = 0.0
    
    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.successes_in_half_open = 0
                log(f"Circuit {self.name}: OPEN -> HALF_OPEN (testing recovery)")
                return True
            return False
        
        # HALF_OPEN - allow limited requests
        return True
    
    def record_success(self):
        """Record successful call."""
        if self.state == "HALF_OPEN":
            self.successes_in_half_open += 1
            if self.successes_in_half_open >= self.half_open_requests:
                self.state = "CLOSED"
                self.failures = 0
                log(f"Circuit {self.name}: HALF_OPEN -> CLOSED (recovered)")
        else:
            self.failures = 0
    
    def record_failure(self, error: str = ""):
        """Record failed call."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
            log(f"Circuit {self.name}: HALF_OPEN -> OPEN (still failing: {error[:100]})")
        elif self.failures >= self.failure_threshold:
            self.state = "OPEN"
            log(f"Circuit {self.name}: CLOSED -> OPEN (threshold reached: {error[:100]})")
    
    def get_state(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "failures": self.failures,
            "last_failure": self.last_failure_time
        }


class HealthMonitor:
    """
    Monitor health of critical services in background.
    
    Monitors:
    - Docker daemon
    - Container status
    - Disk space
    - Filesystem writability
    """
    
    def __init__(self, project: 'Project' = None):
        self.project = project
        self.checks = {}
        self._stop_event = threading.Event()
        self._thread = None
        self.heartbeat_file: Optional[Path] = None
        
        # Circuit breakers for external services
        self.docker_breaker = CircuitBreaker("docker", failure_threshold=3, recovery_timeout=30)
        self.api_breaker = CircuitBreaker("llm_api", failure_threshold=2, recovery_timeout=120)
    
    def start(self, ralph_dir: Path = None):
        """Start background health monitoring."""
        if self._thread and self._thread.is_alive():
            return
        
        if ralph_dir:
            self.heartbeat_file = ralph_dir / ".heartbeat"
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        log("Health monitor started")
    
    def stop(self):
        """Stop health monitoring."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        log("Health monitor stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Update heartbeat
                if self.heartbeat_file:
                    try:
                        self.heartbeat_file.write_text(str(time.time()))
                    except Exception:
                        pass  # Filesystem may be RO
                
                # Docker check
                self._check_docker()
                
                # Container check (if project set)
                if self.project:
                    self._check_container()
                
                # Disk space check
                self._check_disk()
                
            except Exception as e:
                log_error(f"Health check error: {e}")
            
            # Check every 30 seconds
            self._stop_event.wait(30)
    
    def _check_docker(self):
        """Check Docker daemon is responsive."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                self.checks["docker"] = {"healthy": True, "time": time.time()}
                self.docker_breaker.record_success()
            else:
                self.checks["docker"] = {"healthy": False, "error": "docker info failed", "time": time.time()}
                self.docker_breaker.record_failure("docker info failed")
        except subprocess.TimeoutExpired:
            self.checks["docker"] = {"healthy": False, "error": "timeout", "time": time.time()}
            self.docker_breaker.record_failure("timeout")
        except Exception as e:
            self.checks["docker"] = {"healthy": False, "error": str(e), "time": time.time()}
            self.docker_breaker.record_failure(str(e))
    
    def _check_container(self):
        """Check project container is running."""
        if not self.project:
            return
        
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name=^{self.project.container_name}$", "-q"],
                capture_output=True, text=True, timeout=10
            )
            running = bool(result.stdout.strip())
            self.checks["container"] = {
                "healthy": running,
                "name": self.project.container_name,
                "time": time.time()
            }
        except Exception as e:
            self.checks["container"] = {"healthy": False, "error": str(e), "time": time.time()}
    
    def _check_disk(self):
        """Check disk space."""
        try:
            if self.heartbeat_file:
                stat = os.statvfs(self.heartbeat_file.parent)
            else:
                stat = os.statvfs("/")
            free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            self.checks["disk"] = {
                "healthy": free_mb > MIN_FREE_SPACE_MB,
                "free_mb": free_mb,
                "time": time.time()
            }
        except Exception as e:
            self.checks["disk"] = {"healthy": False, "error": str(e), "time": time.time()}
    
    def is_docker_healthy(self) -> bool:
        """Quick check if Docker is healthy (from cache)."""
        check = self.checks.get("docker", {})
        # Consider stale if older than 60 seconds
        if check.get("time", 0) < time.time() - 60:
            self._check_docker()
            check = self.checks.get("docker", {})
        return check.get("healthy", True) and self.docker_breaker.can_proceed()
    
    def is_container_healthy(self) -> bool:
        """Quick check if container is healthy."""
        return self.checks.get("container", {}).get("healthy", True)
    
    def get_status(self) -> dict:
        """Get all health check results."""
        return {
            "checks": self.checks,
            "circuit_breakers": {
                "docker": self.docker_breaker.get_state(),
                "api": self.api_breaker.get_state()
            }
        }


def classify_error(error_msg: str) -> Tuple[str, bool, float]:
    """
    Classify error and return (category, retriable, suggested_delay).
    
    Categories:
    - network: connection issues (retriable)
    - auth: authentication issues (NOT retriable)
    - rate_limit: too many requests (retriable with longer delay)
    - server: API server error (retriable)
    - client: bad request (NOT retriable)
    - timeout: operation timeout (retriable)
    - unknown: unclassified (retriable once)
    """
    error_lower = error_msg.lower()
    
    # Authentication - not retriable (must be actual API auth errors, not code)
    if any(x in error_lower for x in ['litellm.authenticationerror', 'invalid api key', 'unauthorized', 'http 401']):
        return ("auth", False, 0)
    
    # Rate limiting - retriable with long delay
    if any(x in error_lower for x in ['litellm.ratelimiterror', 'rate_limit', 'too many requests', 'http 429']):
        return ("rate_limit", True, 60.0)  # 1 minute
    
    # Network issues - retriable
    if any(x in error_lower for x in ['connection', 'timeout', 'network', 'socket', 'eof', 'reset']):
        return ("network", True, 10.0)
    
    # Server errors - retriable
    if any(x in error_lower for x in ['500', '502', '503', '504', 'internal server', 'service unavailable']):
        return ("server", True, 30.0)
    
    # Client errors - retriable once (might be transient or false positive from code)
    if any(x in error_lower for x in ['litellm.badrequest', 'http 400', 'malformed']):
        return ("client", True, 5.0)
    
    # Timeout - retriable
    if 'timeout' in error_lower:
        return ("timeout", True, 5.0)
    
    # Unknown - retry once with short delay
    return ("unknown", True, 5.0)


class NotificationManager:
    """
    Send notifications for critical events via various channels.
    
    Supported channels:
    - Telegram (via bot API)
    - Slack (via webhook)
    - Discord (via webhook)
    - Email (via SMTP)
    - Desktop (via notify-send on Linux)
    
    Configuration via environment variables or config file.
    """
    
    def __init__(self, ralph_dir: Path = None):
        self.ralph_dir = ralph_dir
        self.config = self._load_config()
        self._last_notification = {}  # Rate limiting
        self._min_interval = 300  # Min 5 minutes between same type
    
    def _load_config(self) -> dict:
        """Load notification config from env or file."""
        config = {
            "telegram_bot_token": os.environ.get("RALPH_TELEGRAM_TOKEN"),
            "telegram_chat_id": os.environ.get("RALPH_TELEGRAM_CHAT"),
            "slack_webhook": os.environ.get("RALPH_SLACK_WEBHOOK"),
            "discord_webhook": os.environ.get("RALPH_DISCORD_WEBHOOK"),
            "email_smtp_server": os.environ.get("RALPH_SMTP_SERVER"),
            "email_smtp_port": os.environ.get("RALPH_SMTP_PORT", "587"),
            "email_from": os.environ.get("RALPH_EMAIL_FROM"),
            "email_to": os.environ.get("RALPH_EMAIL_TO"),
            "email_password": os.environ.get("RALPH_EMAIL_PASSWORD"),
            "desktop_notify": os.environ.get("RALPH_DESKTOP_NOTIFY", "false").lower() == "true",
        }
        
        # Also try config file
        if self.ralph_dir:
            config_file = self.ralph_dir / "notifications.json"
            if config_file.exists():
                try:
                    file_config = json.loads(config_file.read_text())
                    for k, v in file_config.items():
                        if v and not config.get(k):
                            config[k] = v
                except Exception:
                    pass
        
        return config
    
    def is_configured(self) -> bool:
        """Check if any notification channel is configured."""
        return any([
            self.config.get("telegram_bot_token") and self.config.get("telegram_chat_id"),
            self.config.get("slack_webhook"),
            self.config.get("discord_webhook"),
            self.config.get("email_smtp_server") and self.config.get("email_to"),
            self.config.get("desktop_notify"),
        ])
    
    def _rate_limit_ok(self, event_type: str) -> bool:
        """Check if we can send notification (rate limiting)."""
        last = self._last_notification.get(event_type, 0)
        if time.time() - last < self._min_interval:
            return False
        self._last_notification[event_type] = time.time()
        return True
    
    def notify(self, title: str, message: str, level: str = "info", 
               event_type: str = "general") -> bool:
        """
        Send notification via all configured channels.
        
        Args:
            title: Notification title
            message: Notification body
            level: info, warning, error, critical
            event_type: For rate limiting (same type = limited)
        
        Returns:
            True if at least one notification sent
        """
        if not self._rate_limit_ok(event_type):
            return False
        
        sent = False
        
        # Telegram
        if self.config.get("telegram_bot_token") and self.config.get("telegram_chat_id"):
            if self._send_telegram(title, message, level):
                sent = True
        
        # Slack
        if self.config.get("slack_webhook"):
            if self._send_slack(title, message, level):
                sent = True
        
        # Discord
        if self.config.get("discord_webhook"):
            if self._send_discord(title, message, level):
                sent = True
        
        # Desktop
        if self.config.get("desktop_notify"):
            self._send_desktop(title, message, level)
            sent = True
        
        return sent
    
    def _send_telegram(self, title: str, message: str, level: str) -> bool:
        """Send via Telegram Bot API."""
        try:
            import urllib.request
            import urllib.parse
            
            token = self.config["telegram_bot_token"]
            chat_id = self.config["telegram_chat_id"]
            
            emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(level, "📢")
            text = f"{emoji} *{title}*\n\n{message}"
            
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }).encode()
            
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            log_error(f"Telegram notification failed: {e}")
            return False
    
    def _send_slack(self, title: str, message: str, level: str) -> bool:
        """Send via Slack webhook."""
        try:
            import urllib.request
            
            webhook = self.config["slack_webhook"]
            color = {"info": "#36a64f", "warning": "#ffcc00", "error": "#ff0000", "critical": "#8B0000"}.get(level, "#808080")
            
            payload = json.dumps({
                "attachments": [{
                    "color": color,
                    "title": title,
                    "text": message,
                    "ts": int(time.time())
                }]
            }).encode()
            
            req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            log_error(f"Slack notification failed: {e}")
            return False
    
    def _send_discord(self, title: str, message: str, level: str) -> bool:
        """Send via Discord webhook."""
        try:
            import urllib.request
            
            webhook = self.config["discord_webhook"]
            color = {"info": 0x36a64f, "warning": 0xffcc00, "error": 0xff0000, "critical": 0x8B0000}.get(level, 0x808080)
            
            payload = json.dumps({
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": datetime.now().isoformat()
                }]
            }).encode()
            
            req = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 204  # Discord returns 204
        except Exception as e:
            log_error(f"Discord notification failed: {e}")
            return False
    
    def _send_desktop(self, title: str, message: str, level: str):
        """Send desktop notification via notify-send."""
        try:
            # Check if notify-send is available
            if shutil.which("notify-send") is None:
                return
            urgency = {"info": "low", "warning": "normal", "error": "critical", "critical": "critical"}.get(level, "normal")
            subprocess.run(
                ["notify-send", "-u", urgency, title, message],
                capture_output=True, timeout=5
            )
        except Exception:
            pass  # Desktop notifications are best-effort
    
    # Convenience methods for common events
    def notify_iteration_complete(self, iteration: int, result: str, task_id: str = ""):
        """Notify on iteration completion (only for milestones)."""
        if iteration % 10 == 0:  # Every 10 iterations
            self.notify(
                f"Ralph: Iteration {iteration} complete",
                f"Result: {result}\nTask: {task_id}",
                level="info",
                event_type="iteration"
            )
    
    def notify_error(self, error: str, iteration: int = 0):
        """Notify on error."""
        self.notify(
            "Ralph: Error occurred",
            f"Iteration: {iteration}\nError: {error[:500]}",
            level="error",
            event_type="error"
        )
    
    def notify_stuck(self, task_id: str, reason: str):
        """Notify when stuck on task."""
        self.notify(
            f"Ralph: Stuck on task {task_id}",
            f"Reason: {reason}",
            level="warning",
            event_type="stuck"
        )
    
    def notify_complete(self, project_name: str, iterations: int):
        """Notify on project completion."""
        self.notify(
            f"🎉 Ralph: Project {project_name} COMPLETE!",
            f"Total iterations: {iterations}\nAll tasks verified and done!",
            level="info",
            event_type="complete"
        )
    
    def notify_critical(self, message: str):
        """Notify on critical events (immediate, no rate limit)."""
        self._last_notification["critical"] = 0  # Reset rate limit
        self.notify(
            "🚨 Ralph: CRITICAL",
            message,
            level="critical",
            event_type="critical"
        )


# =============================================================================
# DISK SPACE MONITOR
# =============================================================================

class DiskSpaceMonitor:
    """Monitors disk space and triggers cleanup when needed."""
    
    def __init__(self, ralph_dir: Path):
        self.ralph_dir = ralph_dir
    
    def get_free_space_mb(self) -> float:
        """Get free disk space in MB."""
        try:
            stat = os.statvfs(self.ralph_dir)
            free_bytes = stat.f_bavail * stat.f_frsize
            return free_bytes / (1024 * 1024)
        except Exception:
            return float('inf')
    
    def is_low_space(self) -> bool:
        """Check if disk space is low."""
        return self.get_free_space_mb() < MIN_FREE_SPACE_MB
    
    def emergency_cleanup(self) -> bool:
        """Emergency cleanup when disk space is low."""
        try:
            log("Emergency cleanup triggered - low disk space")
            self.rotate_logs(emergency=True)
            return True
        except Exception as e:
            log_error(f"Emergency cleanup failed: {e}")
            return False
    
    def rotate_logs(self, emergency: bool = False) -> dict:
        """Regular log rotation.
        
        Rotates (limited history):
        - ralph.log: truncate to last 1MB (500KB emergency)
        - backups/state_*.json: keep last 10 (5 emergency)
        - test_history.json: keep last 100 entries (50 emergency)
        - iterations/*.log older than 7 days: compress to .gz
        """
        import gzip
        stats = {"truncated": [], "deleted": [], "compressed": []}
        
        # 1. ralph.log - truncate if too large
        max_size = 500_000 if emergency else 1_000_000
        ralph_log = self.ralph_dir / "ralph.log"
        if ralph_log.exists() and ralph_log.stat().st_size > max_size:
            content = ralph_log.read_text()
            safe_write_text(ralph_log, f"...(rotated {time.strftime('%Y-%m-%d %H:%M')})...\n\n" + content[-max_size:])
            stats["truncated"].append("ralph.log")
        
        # 2. backups - keep limited count
        keep_count = 5 if emergency else 10
        backup_dir = self.ralph_dir / "backups"
        if backup_dir.exists():
            for pattern in ["state_*.json", "prd_*.json"]:
                backups = sorted(backup_dir.glob(pattern))
                for old in backups[:-keep_count]:
                    old.unlink()
                    stats["deleted"].append(old.name)
        
        # 3. test_history.json - keep limited entries
        keep_entries = 50 if emergency else 100
        test_history = self.ralph_dir / "test_history.json"
        if test_history.exists():
            try:
                history = json.loads(test_history.read_text())
                if len(history) > keep_entries:
                    safe_write_json(test_history, history[-keep_entries:])
                    stats["truncated"].append(f"test_history ({len(history)}->{keep_entries})")
            except Exception:
                pass
        
        # 4. Compress old iteration logs (older than 7 days)
        iterations_dir = self.ralph_dir / "iterations"
        if iterations_dir.exists():
            cutoff = time.time() - (7 * 24 * 3600)  # 7 days ago
            for log_file in iterations_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff:
                    try:
                        gz_path = log_file.with_suffix(".log.gz")
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(gz_path, 'wb') as f_out:
                                f_out.writelines(f_in)
                        log_file.unlink()
                        stats["compressed"].append(log_file.name)
                    except Exception:
                        pass
        
        return stats



# =============================================================================
# KNOWLEDGE MANAGER - Structured knowledge preservation
# =============================================================================

class KnowledgeManager:
    """
    Manages structured knowledge preservation between iterations.
    
    Key features:
    - Categorized knowledge storage
    - Priority-based retention
    - Automatic summarization of old content
    - Context window management
    """
    
    def __init__(self, ralph_dir: Path, container_name: str = None):
        self.ralph_dir = ralph_dir
        self.container_name = container_name
        self.knowledge_dir = ralph_dir / "knowledge"
        self.critical_file = self.knowledge_dir / "critical.md"
        self.patterns_file = self.knowledge_dir / "patterns.md"
        self.issues_file = self.knowledge_dir / "issues.md"
        self.api_file = self.knowledge_dir / "apis.md"
        self.decisions_file = self.knowledge_dir / "decisions.md"
        self.raw_learnings = ralph_dir / "learnings.md"
        
        # Directory is created by RalphManager via Docker
    
    def _write_file(self, subpath: str, content: str) -> bool:
        """Write file via Docker if container available."""
        if self.container_name and Docker.container_running(self.container_name):
            return Docker.write_file(self.container_name, f"/workspace/.ralph/{subpath}", content)
        return safe_write_text(self.ralph_dir / subpath, content)
    
    def _read_file(self, subpath: str) -> str:
        """Read file via Docker if container available."""
        if self.container_name and Docker.container_running(self.container_name):
            return Docker.read_file(self.container_name, f"/workspace/.ralph/{subpath}") or ""
        path = self.ralph_dir / subpath
        return path.read_text() if path.exists() else ""
    
    def _append_file(self, subpath: str, content: str) -> bool:
        """Append to file via Docker if container available."""
        if self.container_name and Docker.container_running(self.container_name):
            return Docker.append_file(self.container_name, f"/workspace/.ralph/{subpath}", content)
        return safe_append_text(self.ralph_dir / subpath, content)
    
    def init_structure(self):
        """Initialize knowledge structure."""
        # Critical knowledge - always preserved
        if not self.critical_file.exists():
            self.critical_file.write_text("""# Critical Knowledge

> This file contains CRITICAL information that MUST NOT be lost.
> Only add information that is essential for project success.

## Project Architecture

(Added by Architect iterations)

## Critical Patterns

(Patterns that must be followed)

## Critical Issues to Avoid

(Issues that caused major problems)

## External API Details

(API endpoints, authentication, etc.)

## Build/Deploy Requirements

(Essential build steps)
""")
        
        # Patterns knowledge
        if not self.patterns_file.exists():
            self.patterns_file.write_text("""# Code Patterns

## Successful Patterns

(Patterns that worked well)

## Anti-Patterns

(Patterns to avoid)

## Refactoring Patterns

(How to safely refactor)
""")
        
        # Issues knowledge
        if not self.issues_file.exists():
            self.issues_file.write_text("""# Known Issues

## Resolved Issues

(Issues that were fixed - include solution)

## Open Issues

(Issues still being worked on)

## Workarounds

(Temporary solutions)
""")
        
        # API knowledge
        if not self.api_file.exists():
            self.api_file.write_text("""# External APIs & Services

## Service: [Name]

- Endpoint: 
- Auth: 
- Rate limits: 
- Key methods: 

## Service: [Name]

...
""")
        
        # Decisions knowledge
        if not self.decisions_file.exists():
            self.decisions_file.write_text("""# Architecture Decisions

## Decision: [Title]

- Date: 
- Context: 
- Decision: 
- Consequences: 

## Decision: [Title]

...
""")
    
    def add_critical(self, category: str, content: str):
        """Add critical knowledge that must be preserved."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n### [{timestamp}] {category}\n{content}\n"
        self._append_file("knowledge/critical.md", entry)
    
    def add_pattern(self, name: str, description: str, is_anti: bool = False):
        """Add a code pattern."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        pattern_type = "Anti-Pattern" if is_anti else "Pattern"
        entry = f"\n### [{timestamp}] {name} ({pattern_type})\n{description}\n"
        self._append_file("knowledge/patterns.md", entry)
    
    def add_issue(self, issue: str, solution: str = "", status: str = "open"):
        """Add an issue with optional solution."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        entry = f"\n### [{timestamp}] {issue}\n"
        entry += f"- Status: {status}\n"
        if solution:
            entry += f"- Solution: {solution}\n"
        entry += "\n"
        self._append_file("knowledge/issues.md", entry)
    
    def add_decision(self, title: str, context: str, decision: str, consequences: str = ""):
        """Add an architecture decision."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        entry = f"\n### [{timestamp}] {title}\n"
        entry += f"- Context: {context}\n"
        entry += f"- Decision: {decision}\n"
        if consequences:
            entry += f"- Consequences: {consequences}\n"
        entry += "\n"
        self._append_file("knowledge/decisions.md", entry)
    
    def get_context(self, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
        """
        Get knowledge context optimized for token budget.
        
        Priority order:
        1. Critical knowledge (always included)
        2. Recent patterns
        3. Recent issues
        4. API docs (if space permits)
        5. Decisions (if space permits)
        """
        parts = []
        remaining_chars = max_tokens * CHARS_PER_TOKEN
        
        # Always include critical knowledge (truncated if needed)
        critical = self._read_file("knowledge/critical.md")
        if critical:
            if len(critical) > CRITICAL_LEARNINGS_LIMIT:
                # Keep header and last portion
                lines = critical.split('\n')
                header_end = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('#'):
                        header_end = i
                        break
                header = '\n'.join(lines[:header_end + 5])
                recent = critical[-CRITICAL_LEARNINGS_LIMIT:]
                critical = header + "\n\n...(truncated)...\n\n" + recent
            parts.append(critical)
            remaining_chars -= len(critical)
        
        # Add patterns if space permits
        patterns = self._read_file("knowledge/patterns.md")
        if remaining_chars > 5000 and patterns:
            patterns = self._truncate_content(patterns, RECENT_LEARNINGS_LIMIT)
            parts.append("\n## Code Patterns\n" + patterns)
            remaining_chars -= len(patterns) + 20
        
        # Add issues if space permits
        issues = self._read_file("knowledge/issues.md")
        if remaining_chars > 5000 and issues:
            issues = self._truncate_content(issues, ARCHITECTURE_LIMIT)
            parts.append("\n## Known Issues\n" + issues)
            remaining_chars -= len(issues) + 20
        
        # Add API docs if space permits
        apis = self._read_file("knowledge/apis.md")
        if remaining_chars > 8000 and apis:
            apis = self._truncate_content(apis, GUARDRAILS_LIMIT)
            parts.append("\n## External APIs\n" + apis)
            remaining_chars -= len(apis) + 20
        
        # Add decisions if space permits
        decisions = self._read_file("knowledge/decisions.md")
        if remaining_chars > 5000 and decisions:
            decisions = self._truncate_content(decisions, GUARDRAILS_LIMIT)
            parts.append("\n## Architecture Decisions\n" + decisions)
        
        return "\n\n".join(parts)
    
    def _truncate_content(self, content: str, max_chars: int) -> str:
        """Truncate content keeping most recent."""
        if len(content) > max_chars:
            return "...(older content)...\n" + content[-max_chars:]
        return content
    
    def consolidate(self):
        """
        Consolidate old knowledge by summarizing.
        Called periodically to prevent unbounded growth.
        """
        self._consolidate_learnings()
        self._consolidate_knowledge_files()
        self._cleanup_old_archives()
    
    def _consolidate_learnings(self):
        """Consolidate raw learnings file."""
        if not self.raw_learnings.exists():
            return
            
        content = self.raw_learnings.read_text()
        
        # If very large, archive and keep summary
        if len(content) > 80000:  # ~20K tokens
            lines = content.split('\n')
            
            if len(lines) > 2000:
                # Extract key learnings (lines starting with - or * or containing keywords)
                key_patterns = ['critical', 'important', 'error', 'fix', 'solution', 'pattern', 'avoid']
                key_learnings = []
                for line in lines[:-1500]:  # From old content
                    line_lower = line.lower()
                    if any(p in line_lower for p in key_patterns) or line.strip().startswith(('- ', '* ', '### ')):
                        key_learnings.append(line)
                
                # Archive old content
                archive_file = self.knowledge_dir / f"learnings_archive_{int(time.time())}.md"
                archive_content = '\n'.join(lines[:-1500])
                safe_write_text(archive_file, archive_content)
                
                # Keep recent + extracted key learnings
                recent = '\n'.join(lines[-1500:])
                key_summary = '\n'.join(key_learnings[-200:]) if key_learnings else ""
                
                safe_write_text(
                    self.raw_learnings,
                    f"# Learnings\n\n"
                    f"(Old content archived to {archive_file.name})\n\n"
                    f"## Key Points from Archive\n{key_summary}\n\n"
                    f"## Recent\n{recent}"
                )
                log(f"Consolidated learnings: archived {len(lines)-1500} lines, kept {len(key_learnings)} key points")
    
    def _consolidate_knowledge_files(self):
        """Consolidate individual knowledge files when too large."""
        # Consolidate critical.md
        if self.critical_file.exists():
            content = self.critical_file.read_text()
            if len(content) > CRITICAL_LEARNINGS_LIMIT * 1.5:
                # Keep header and deduplicate entries
                lines = content.split('\n')
                header_end = 0
                for i, line in enumerate(lines):
                    if line.startswith('## ') and i > 5:
                        header_end = i
                        break
                
                header = '\n'.join(lines[:header_end]) if header_end else lines[:20]
                body_lines = lines[header_end:] if header_end else lines[20:]
                
                # Deduplicate similar entries
                unique_entries = []
                seen_hashes = set()
                current_entry = []
                
                for line in body_lines:
                    if line.startswith('### '):
                        if current_entry:
                            entry_text = '\n'.join(current_entry)
                            entry_hash = hash(entry_text[:100])  # Hash first 100 chars
                            if entry_hash not in seen_hashes:
                                unique_entries.extend(current_entry)
                                seen_hashes.add(entry_hash)
                        current_entry = [line]
                    else:
                        current_entry.append(line)
                
                if current_entry:
                    unique_entries.extend(current_entry)
                
                # Keep only recent if still too large
                if len(unique_entries) > 1500:
                    unique_entries = unique_entries[-1500:]
                
                new_content = '\n'.join(header) + '\n\n' + '\n'.join(unique_entries)
                safe_write_text(self.critical_file, new_content)
                log(f"Consolidated critical.md: {len(content)} -> {len(new_content)} chars")
    
    def _cleanup_old_archives(self):
        """Remove old archive files, keep only recent ones."""
        archives = sorted(self.knowledge_dir.glob("*_archive_*.md"))
        if len(archives) > 10:
            for old in archives[:-10]:
                old.unlink()
                log(f"Removed old archive: {old.name}")


# =============================================================================
# CONTEXT WINDOW MANAGER
# =============================================================================

class ContextWindowManager:
    """
    Manages prompt size to stay within token limits.
    
    Tracks estimated token usage and optimizes content inclusion.
    """
    
    def __init__(self, max_tokens: int = MAX_PROMPT_TOKENS):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * CHARS_PER_TOKEN
        self.usage_history: List[int] = []  # Track prompt sizes
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return len(text) // CHARS_PER_TOKEN
    
    def optimize_prompt(self, sections: Dict[str, str], priorities: Dict[str, int]) -> str:
        """
        Optimize prompt to fit within token budget.
        
        Args:
            sections: Dict of section_name -> content
            priorities: Dict of section_name -> priority (higher = more important)
        
        Returns:
            Optimized prompt string
        """
        # Sort sections by priority (descending)
        sorted_sections = sorted(
            sections.items(),
            key=lambda x: priorities.get(x[0], 50),
            reverse=True
        )
        
        result_parts = []
        remaining_chars = self.max_chars
        
        for name, content in sorted_sections:
            if remaining_chars <= 0:
                break
            
            content_len = len(content)
            
            if content_len <= remaining_chars:
                # Full content fits
                result_parts.append(content)
                remaining_chars -= content_len
            else:
                # Need to truncate
                available = remaining_chars - 100  # Leave buffer
                if available > 500:
                    truncated = content[-available:] if name in ['learnings', 'architecture'] else content[:available]
                    result_parts.append(f"...(truncated)...\n{truncated}")
                    remaining_chars = 0
        
        return "\n\n".join(result_parts)
    
    def log_usage(self, prompt: str):
        """Log prompt size for tracking."""
        tokens = self.estimate_tokens(prompt)
        self.usage_history.append(tokens)
        
        # Keep last 100 entries
        if len(self.usage_history) > 100:
            self.usage_history = self.usage_history[-100:]
    
    def get_stats(self) -> dict:
        """Get context usage statistics."""
        if not self.usage_history:
            return {"avg": 0, "max": 0, "min": 0, "count": 0}
        
        return {
            "avg": sum(self.usage_history) / len(self.usage_history),
            "max": max(self.usage_history),
            "min": min(self.usage_history),
            "count": len(self.usage_history),
            "recent_avg": sum(self.usage_history[-10:]) / min(10, len(self.usage_history[-10:]))
        }


# =============================================================================
# GIT QUALITY GATE - Automatic code quality checks via git
# =============================================================================

class GitQualityGate:
    """
    Automatic quality checks using git.
    
    Features:
    - Diff analysis before commit
    - Automatic rollback on failures
    - Change size limits
    - Suspicious pattern detection
    - Task verification via git changes
    - Per-task commit tracking
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.max_lines_per_task = 500  # Max lines changed per task
        self.max_files_per_task = 20   # Max files changed per task
        
        # Suspicious patterns that should create cleanup tasks
        self.suspicious_patterns = [
            (r'TODO|FIXME|XXX|HACK', 'cleanup', 'Remove TODO/FIXME comments'),
            (r'print\(|console\.log', 'cleanup', 'Remove debug prints'),
            (r'password\s*=\s*["\'][^"\']+["\']', 'security', 'Hardcoded password detected'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'security', 'Hardcoded API key detected'),
        ]
        
        # Track task start commits for verification
        self._task_start_commits: Dict[str, str] = {}
    
    def check_diff(self) -> Tuple[bool, str, List[dict]]:
        """
        Check git diff quality before commit.
        
        Returns:
            (passed, message, issues_found)
        """
        issues = []
        
        try:
            # Get diff stats
            result = subprocess.run(
                ['git', 'diff', '--stat', '--cached'],
                capture_output=True, text=True, cwd=self.workspace, timeout=30
            )
            
            if result.returncode != 0:
                # Try unstaged diff
                result = subprocess.run(
                    ['git', 'diff', '--stat'],
                    capture_output=True, text=True, cwd=self.workspace, timeout=30
                )
            
            # Parse stats
            lines_added, lines_removed, files_changed = self._parse_diff_stats(result.stdout)
            total_lines = lines_added + lines_removed
            
            # Check limits
            if total_lines > self.max_lines_per_task:
                issues.append({
                    'type': 'size',
                    'severity': 'warning',
                    'message': f'Too many lines changed: {total_lines} (max: {self.max_lines_per_task})'
                })
            
            if files_changed > self.max_files_per_task:
                issues.append({
                    'type': 'size',
                    'severity': 'warning',
                    'message': f'Too many files changed: {files_changed} (max: {self.max_files_per_task})'
                })
            
            # Get full diff for pattern check
            full_diff = subprocess.run(
                ['git', 'diff'],
                capture_output=True, text=True, cwd=self.workspace, timeout=30
            )
            
            # Check suspicious patterns
            for pattern, issue_type, description in self.suspicious_patterns:
                matches = re.findall(pattern, full_diff.stdout, re.IGNORECASE)
                if matches:
                    issues.append({
                        'type': issue_type,
                        'severity': 'info' if issue_type == 'cleanup' else 'warning',
                        'message': f'{description} ({len(matches)} occurrences)',
                        'pattern': pattern
                    })
            
            # Determine pass/fail
            critical_issues = [i for i in issues if i['severity'] == 'error']
            if critical_issues:
                return False, critical_issues[0]['message'], issues
            
            return True, f"OK: {total_lines} lines in {files_changed} files", issues
            
        except subprocess.TimeoutExpired:
            return False, "Git diff timed out", []
        except Exception as e:
            return True, f"Could not check diff: {e}", []  # Pass on error to not block
    
    def _parse_diff_stats(self, stats_output: str) -> Tuple[int, int, int]:
        """Parse git diff --stat output."""
        lines_added = 0
        lines_removed = 0
        files_changed = 0
        
        for line in stats_output.split('\n'):
            # Match lines like: " file.py | 10 ++++----"
            match = re.search(r'\|\s*(\d+)\s*([+-]*)', line)
            if match:
                files_changed += 1
                changes = int(match.group(1))
                symbols = match.group(2)
                plus_count = symbols.count('+')
                minus_count = symbols.count('-')
                total = plus_count + minus_count
                if total > 0:
                    lines_added += int(changes * plus_count / total)
                    lines_removed += int(changes * minus_count / total)
        
        return lines_added, lines_removed, files_changed
    
    def rollback_last_commit(self) -> bool:
        """Rollback the last commit (soft reset)."""
        try:
            result = subprocess.run(
                ['git', 'reset', '--soft', 'HEAD~1'],
                capture_output=True, text=True, cwd=self.workspace, timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def rollback_to_commit(self, commit_hash: str) -> bool:
        """Rollback to a specific commit."""
        try:
            result = subprocess.run(
                ['git', 'reset', '--hard', commit_hash],
                capture_output=True, text=True, cwd=self.workspace, timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_last_good_commit(self) -> Optional[str]:
        """Find the last commit before Ralph started working."""
        try:
            result = subprocess.run(
                ['git', 'log', '--oneline', '-50'],
                capture_output=True, text=True, cwd=self.workspace, timeout=30
            )
            for line in result.stdout.split('\n'):
                if '[Ralph]' not in line and 'TASK-' not in line:
                    # Found a non-Ralph commit
                    return line.split()[0] if line else None
            return None
        except Exception:
            return None
    
    # =========================================================================
    # GIT-BASED TASK VERIFICATION
    # =========================================================================
    
    def is_git_repo(self) -> bool:
        """Check if workspace is a git repository."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                capture_output=True, cwd=self.workspace, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_current_commit(self) -> Optional[str]:
        """Get current HEAD commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=self.workspace, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def mark_task_start(self, task_id: str):
        """Record commit hash when task starts (for later verification)."""
        commit = self.get_current_commit()
        if commit:
            self._task_start_commits[task_id] = commit
            log(f"Git: Task {task_id} starting from commit {commit[:8]}")
    
    def verify_task_changes(self, task_id: str) -> Tuple[bool, str, dict]:
        """
        Verify that a task actually made changes in git.
        
        Returns:
            (has_changes, message, details)
        """
        details = {
            "task_id": task_id,
            "has_changes": False,
            "files_changed": [],
            "lines_added": 0,
            "lines_removed": 0,
            "commits_made": 0
        }
        
        if not self.is_git_repo():
            return True, "Not a git repo - skipping verification", details
        
        start_commit = self._task_start_commits.get(task_id)
        if not start_commit:
            # No start commit recorded, check for any uncommitted changes
            return self._check_uncommitted_changes(details)
        
        try:
            # Check if there are new commits since task start
            result = subprocess.run(
                ['git', 'log', '--oneline', f'{start_commit}..HEAD'],
                capture_output=True, text=True, cwd=self.workspace, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                commits = result.stdout.strip().split('\n')
                details["commits_made"] = len(commits)
                details["has_changes"] = True
                
                # Get diff stats
                diff_result = subprocess.run(
                    ['git', 'diff', '--stat', start_commit, 'HEAD'],
                    capture_output=True, text=True, cwd=self.workspace, timeout=10
                )
                
                if diff_result.returncode == 0:
                    added, removed, files = self._parse_diff_stats(diff_result.stdout)
                    details["lines_added"] = added
                    details["lines_removed"] = removed
                    details["files_changed"] = self._get_changed_files(start_commit)
                
                return True, f"Task made {len(commits)} commit(s), {added}+ {removed}- lines", details
            
            # No commits, check for uncommitted changes
            return self._check_uncommitted_changes(details)
            
        except Exception as e:
            return True, f"Verification error: {e}", details
    
    def _check_uncommitted_changes(self, details: dict) -> Tuple[bool, str, dict]:
        """Check for uncommitted changes (staged or unstaged)."""
        try:
            # Check for any changes (staged or not)
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, cwd=self.workspace, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                changes = result.stdout.strip().split('\n')
                details["has_changes"] = True
                details["files_changed"] = [c[3:] for c in changes if len(c) > 3]
                
                # Get line counts from diff
                diff_result = subprocess.run(
                    ['git', 'diff', '--stat'],
                    capture_output=True, text=True, cwd=self.workspace, timeout=10
                )
                if diff_result.returncode == 0:
                    added, removed, _ = self._parse_diff_stats(diff_result.stdout)
                    details["lines_added"] = added
                    details["lines_removed"] = removed
                
                return True, f"Uncommitted changes: {len(changes)} files", details
            
            details["has_changes"] = False
            return False, "No git changes detected - task may not have completed work", details
            
        except Exception as e:
            return True, f"Check error: {e}", details
    
    def _get_changed_files(self, since_commit: str) -> List[str]:
        """Get list of files changed since a commit."""
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', since_commit, 'HEAD'],
                capture_output=True, text=True, cwd=self.workspace, timeout=10
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.split('\n') if f.strip()]
            return []
        except Exception:
            return []
    
    def get_task_diff(self, task_id: str) -> Optional[str]:
        """Get the full diff for a task (since task start)."""
        start_commit = self._task_start_commits.get(task_id)
        if not start_commit:
            # Return current uncommitted diff
            try:
                result = subprocess.run(
                    ['git', 'diff'],
                    capture_output=True, text=True, cwd=self.workspace, timeout=30
                )
                return result.stdout if result.returncode == 0 else None
            except Exception:
                return None
        
        try:
            result = subprocess.run(
                ['git', 'diff', start_commit, 'HEAD'],
                capture_output=True, text=True, cwd=self.workspace, timeout=30
            )
            return result.stdout if result.returncode == 0 else None
        except Exception:
            return None
    
    def create_task_commit(self, task_id: str, message: str = None) -> bool:
        """
        Create a commit for completed task with all current changes.
        
        Commits all staged and unstaged changes with task ID in message.
        """
        if not self.is_git_repo():
            return False
        
        try:
            # Stage all changes
            subprocess.run(
                ['git', 'add', '-A'],
                capture_output=True, cwd=self.workspace, timeout=10
            )
            
            # Check if there's anything to commit
            status = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, cwd=self.workspace, timeout=10
            )
            
            if not status.stdout.strip():
                log(f"Git: No changes to commit for {task_id}")
                return False
            
            # Create commit
            commit_msg = message or f"[Ralph] Complete {task_id}"
            result = subprocess.run(
                ['git', 'commit', '-m', commit_msg],
                capture_output=True, text=True, cwd=self.workspace, timeout=30
            )
            
            if result.returncode == 0:
                log(f"Git: Committed {task_id}")
                return True
            else:
                log_error(f"Git commit failed: {result.stderr}")
                return False
                
        except Exception as e:
            log_error(f"Git commit error: {e}")
            return False
    
    def get_recent_commits(self, count: int = 10) -> List[dict]:
        """Get recent commits with details."""
        try:
            result = subprocess.run(
                ['git', 'log', f'-{count}', '--format=%H|%s|%ai|%an'],
                capture_output=True, text=True, cwd=self.workspace, timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        commits.append({
                            "hash": parts[0][:8],
                            "message": parts[1],
                            "date": parts[2],
                            "author": parts[3],
                            "is_ralph": "[Ralph]" in parts[1] or "TASK-" in parts[1]
                        })
            return commits
        except Exception:
            return []


# =============================================================================
# GIT HANDOFF - Worker communication via git
# =============================================================================

class GitHandoff:
    """
    Git-based communication between workers.
    
    Instead of separate files, uses git for:
    - Task context passing via commit messages/notes
    - FIX task context in commit history
    - Worker handoff messages
    - Task completion evidence
    
    Benefits over files:
    - Versioned history of all communications
    - Atomic operations
    - Easy rollback
    - Built-in blame/history
    """
    
    def __init__(self, workspace: Path, ralph_dir: Path):
        self.workspace = workspace
        self.ralph_dir = ralph_dir
        self.handoff_file = workspace / ".ralph" / "handoff.json"
        self.context_file = workspace / ".ralph" / "task_context.json"
        
    def _ensure_ralph_dir(self):
        """Ensure .ralph directory exists in workspace."""
        # Directory should be created by RalphManager via Docker
        # This is a no-op fallback
        pass
        
        # Add to gitignore if not present (optional - could also track it)
        gitignore = self.workspace / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            if ".ralph/" not in content:
                # Don't ignore - we WANT to track handoff in git
                pass
    
    def write_handoff(self, task_id: str, message: str, context: dict = None, 
                      for_next: bool = True) -> bool:
        """
        Write handoff message for next worker.
        
        Args:
            task_id: Current task ID
            message: Handoff message
            context: Additional context dict
            for_next: If True, intended for next iteration
        
        Returns:
            True if successful
        """
        self._ensure_ralph_dir()
        
        handoff = {
            "task_id": task_id,
            "message": message,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "for_next_iteration": for_next
        }
        
        try:
            # Write handoff file
            safe_write_json(self.handoff_file, handoff)
            
            # Commit it so next worker sees it
            self._commit_handoff(task_id, message)
            return True
        except Exception as e:
            log_error(f"Failed to write handoff: {e}")
            return False
    
    def read_handoff(self) -> Optional[dict]:
        """Read handoff from previous worker."""
        if not self.handoff_file.exists():
            return None
        
        try:
            return json.loads(self.handoff_file.read_text())
        except Exception:
            return None
    
    def clear_handoff(self):
        """Clear handoff after reading."""
        if self.handoff_file.exists():
            try:
                self.handoff_file.unlink()
            except Exception:
                pass
    
    def _commit_handoff(self, task_id: str, message: str):
        """Commit handoff file to git."""
        try:
            # Stage handoff file
            subprocess.run(
                ['git', 'add', str(self.handoff_file)],
                capture_output=True, cwd=self.workspace, timeout=5
            )
            
            # Commit with task context
            commit_msg = f"[Ralph:Handoff] {task_id}: {message[:50]}"
            subprocess.run(
                ['git', 'commit', '-m', commit_msg, '--allow-empty'],
                capture_output=True, cwd=self.workspace, timeout=10
            )
        except Exception:
            pass  # Non-critical
    
    def save_task_context(self, task_id: str, context: dict) -> bool:
        """
        Save task context that persists across iterations.
        
        Context includes:
        - What was tried
        - What worked/didn't work
        - Dependencies discovered
        - Notes for future reference
        """
        self._ensure_ralph_dir()
        
        try:
            # Load existing context
            existing = {}
            if self.context_file.exists():
                try:
                    existing = json.loads(self.context_file.read_text())
                except Exception:
                    existing = {}
            
            # Update context for this task
            existing[task_id] = {
                **context,
                "updated": datetime.now().isoformat()
            }
            
            # Keep only recent tasks (max 50)
            if len(existing) > 50:
                # Sort by update time, keep newest
                sorted_tasks = sorted(
                    existing.items(),
                    key=lambda x: x[1].get("updated", ""),
                    reverse=True
                )
                existing = dict(sorted_tasks[:50])
            
            safe_write_json(self.context_file, existing)
            return True
        except Exception as e:
            log_error(f"Failed to save task context: {e}")
            return False
    
    def get_task_context(self, task_id: str) -> Optional[dict]:
        """Get saved context for a task."""
        if not self.context_file.exists():
            return None
        
        try:
            all_context = json.loads(self.context_file.read_text())
            return all_context.get(task_id)
        except Exception:
            return None
    
    def get_fix_context(self, original_task_id: str) -> List[dict]:
        """
        Get context from previous FIX attempts for same original task.
        
        Uses git history to find what was tried before.
        """
        fix_attempts = []
        
        try:
            # Search git log for commits related to original task
            result = subprocess.run(
                ['git', 'log', '--oneline', '--grep', f'FIX-{original_task_id}', '-20'],
                capture_output=True, text=True, cwd=self.workspace, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(' ', 1)
                        fix_attempts.append({
                            "commit": parts[0],
                            "message": parts[1] if len(parts) > 1 else ""
                        })
            
            # Also check task_context for recorded attempts
            context = self.get_task_context(f"FIX-{original_task_id}")
            if context:
                fix_attempts.append({
                    "type": "context",
                    "data": context
                })
            
            return fix_attempts
        except Exception:
            return []
    
    def record_fix_attempt(self, fix_task_id: str, original_error: str, 
                          approach: str, result: str) -> bool:
        """
        Record a FIX task attempt for future reference.
        
        Helps avoid repeating failed approaches.
        """
        context = {
            "original_error": original_error[:500],
            "approach": approach,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get existing attempts
        existing = self.get_task_context(fix_task_id) or {}
        attempts = existing.get("attempts", [])
        attempts.append(context)
        existing["attempts"] = attempts[-10:]  # Keep last 10 attempts
        
        return self.save_task_context(fix_task_id, existing)
    
    def create_git_note(self, task_id: str, note: str) -> bool:
        """
        Add git note to current HEAD.
        
        Git notes are metadata attached to commits, separate from commit message.
        Useful for adding context without changing commit history.
        """
        try:
            result = subprocess.run(
                ['git', 'notes', 'add', '-m', f"[{task_id}] {note}"],
                capture_output=True, text=True, cwd=self.workspace, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_git_notes(self, commit: str = "HEAD") -> Optional[str]:
        """Get git notes for a commit."""
        try:
            result = subprocess.run(
                ['git', 'notes', 'show', commit],
                capture_output=True, text=True, cwd=self.workspace, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def tag_task_complete(self, task_id: str, summary: str = "") -> bool:
        """
        Create git tag for completed task.
        
        Allows easy rollback to state after specific task.
        """
        try:
            tag_name = f"ralph/{task_id.lower()}"
            msg = summary or f"Completed {task_id}"
            
            result = subprocess.run(
                ['git', 'tag', '-a', tag_name, '-m', msg],
                capture_output=True, text=True, cwd=self.workspace, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_task_tags(self) -> List[dict]:
        """Get all Ralph task tags."""
        try:
            result = subprocess.run(
                ['git', 'tag', '-l', 'ralph/*'],
                capture_output=True, text=True, cwd=self.workspace, timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            tags = []
            for tag in result.stdout.strip().split('\n'):
                if tag:
                    # Get tag message
                    msg_result = subprocess.run(
                        ['git', 'tag', '-l', tag, '-n1'],
                        capture_output=True, text=True, cwd=self.workspace, timeout=5
                    )
                    tags.append({
                        "tag": tag,
                        "task_id": tag.replace("ralph/", "").upper(),
                        "message": msg_result.stdout.strip() if msg_result.returncode == 0 else ""
                    })
            return tags
        except Exception:
            return []
    
    def rollback_to_task(self, task_id: str) -> bool:
        """Rollback to state after a specific task completed."""
        tag_name = f"ralph/{task_id.lower()}"
        try:
            result = subprocess.run(
                ['git', 'reset', '--hard', tag_name],
                capture_output=True, cwd=self.workspace, timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_blame_for_file(self, filepath: str) -> List[dict]:
        """
        Get git blame for file to see which task changed each line.
        
        Useful for understanding where bugs were introduced.
        """
        try:
            result = subprocess.run(
                ['git', 'blame', '--line-porcelain', filepath],
                capture_output=True, text=True, cwd=self.workspace, timeout=30
            )
            
            if result.returncode != 0:
                return []
            
            # Parse blame output
            blame_info = []
            current = {}
            
            for line in result.stdout.split('\n'):
                if line.startswith('author '):
                    current['author'] = line[7:]
                elif line.startswith('summary '):
                    current['summary'] = line[8:]
                    # Check if Ralph commit
                    current['is_ralph'] = '[Ralph]' in line or 'TASK-' in line
                elif line.startswith('\t'):
                    current['code'] = line[1:]
                    blame_info.append(current)
                    current = {}
            
            return blame_info
        except Exception:
            return []


# =============================================================================
# TEST ENFORCEMENT - Automatic test running and fix task creation
# =============================================================================

class TestEnforcement:
    """
    Enforces testing and creates fix tasks automatically.
    
    Features:
    - Detect test framework
    - Run tests and parse results (inside container!)
    - Create FIX tasks for failures
    - Track test coverage trends
    """
    
    def __init__(self, workspace: Path, ralph_dir: Path, container_name: str = None):
        self.workspace = workspace
        self.ralph_dir = ralph_dir
        self.container_name = container_name
        self.test_history_file = ralph_dir / "test_history.json"
        
        # Test commands by framework (run inside container)
        self.test_commands = {
            'pytest': 'pytest -v --tb=short -x',
            'jest': 'npm test -- --passWithNoTests',
            'go': 'go test -v ./...',
            'cargo': 'cargo test',
        }
    
    def detect_framework(self) -> Optional[str]:
        """Detect which test framework is used."""
        if (self.workspace / 'pytest.ini').exists() or \
           (self.workspace / 'pyproject.toml').exists() or \
           list(self.workspace.glob('**/test_*.py')) or \
           list(self.workspace.glob('**/*_test.py')):
            return 'pytest'
        
        if (self.workspace / 'package.json').exists():
            try:
                pkg = json.loads((self.workspace / 'package.json').read_text())
                if 'jest' in pkg.get('devDependencies', {}) or \
                   'jest' in pkg.get('dependencies', {}):
                    return 'jest'
            except Exception:
                pass
        
        if (self.workspace / 'go.mod').exists():
            return 'go'
        
        if (self.workspace / 'Cargo.toml').exists():
            return 'cargo'
        
        return None
    
    def run_tests(self, timeout: int = 300) -> dict:
        """
        Run tests INSIDE container and return results.
        
        IMPORTANT: Tests run inside container via docker exec, not on host!
        This ensures tools like cargo, npm, go are available.
        
        Returns:
            {
                'passed': bool,
                'total': int,
                'failures': int,
                'errors': List[str],
                'output': str
            }
        """
        framework = self.detect_framework()
        if not framework:
            return {
                'passed': True,
                'total': 0,
                'failures': 0,
                'errors': [],
                'output': 'No test framework detected',
                'framework': None
            }
        
        test_cmd = self.test_commands.get(framework, '')
        if not test_cmd:
            return {
                'passed': True,
                'total': 0,
                'failures': 0,
                'errors': [],
                'output': f'Unknown framework: {framework}',
                'framework': framework
            }
        
        # Check if container is available
        if not self.container_name:
            return {
                'passed': True,  # Don't fail if no container - just skip
                'total': 0,
                'failures': 0,
                'errors': [],
                'output': 'No container configured for testing',
                'framework': framework
            }
        
        # Check container is running
        if not Docker.container_running(self.container_name):
            return {
                'passed': True,  # Don't fail if container not running
                'total': 0,
                'failures': 0,
                'errors': [],
                'output': 'Container not running',
                'framework': framework
            }
        
        try:
            # Run tests INSIDE container via docker exec
            # Use bash to ensure PATH includes cargo, npm, etc.
            docker_cmd = [
                'docker', 'exec', '-w', '/workspace',
                self.container_name,
                'bash', '-c',
                f'export PATH="/root/.cargo/bin:/root/.local/bin:$PATH" && {test_cmd}'
            ]
            
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout + '\n' + result.stderr
            passed, total, failures, errors = self._parse_test_output(framework, output, result.returncode)
            
            # Save to history
            self._save_history(passed, total, failures)
            
            return {
                'passed': passed,
                'total': total,
                'failures': failures,
                'errors': errors,
                'output': output[-5000:],  # Limit output size
                'framework': framework,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'total': 0,
                'failures': 1,
                'errors': ['Test execution timed out'],
                'output': f'Timeout after {timeout}s',
                'framework': framework
            }
        except Exception as e:
            # Don't create FIX tasks for infrastructure errors (docker not available, etc.)
            error_str = str(e)
            if 'No such file or directory' in error_str or 'command not found' in error_str:
                return {
                    'passed': True,  # Treat as skip, not failure
                    'total': 0,
                    'failures': 0,
                    'errors': [],
                    'output': f'Test infrastructure unavailable: {error_str}',
                    'framework': framework
                }
            return {
                'passed': False,
                'total': 0,
                'failures': 1,
                'errors': [error_str],
                'output': error_str,
                'framework': framework
            }
    
    def _parse_test_output(self, framework: str, output: str, returncode: int) -> Tuple[bool, int, int, List[str]]:
        """Parse test output to extract results."""
        errors = []
        
        if framework == 'pytest':
            # Look for "X passed, Y failed"
            match = re.search(r'(\d+) passed', output)
            passed_count = int(match.group(1)) if match else 0
            
            match = re.search(r'(\d+) failed', output)
            failed_count = int(match.group(1)) if match else 0
            
            match = re.search(r'(\d+) error', output)
            error_count = int(match.group(1)) if match else 0
            
            # Extract failure messages
            for match in re.finditer(r'FAILED\s+(\S+)', output):
                errors.append(match.group(1))
            
            total = passed_count + failed_count + error_count
            return failed_count == 0 and error_count == 0, total, failed_count + error_count, errors
        
        elif framework == 'jest':
            match = re.search(r'Tests:\s+(\d+) passed,?\s*(\d+)? failed?', output)
            if match:
                passed_count = int(match.group(1))
                failed_count = int(match.group(2)) if match.group(2) else 0
                return failed_count == 0, passed_count + failed_count, failed_count, errors
        
        elif framework == 'go':
            # Count PASS and FAIL
            passed_count = len(re.findall(r'^ok\s+', output, re.MULTILINE))
            failed_count = len(re.findall(r'^FAIL\s+', output, re.MULTILINE))
            return failed_count == 0, passed_count + failed_count, failed_count, errors
        
        elif framework == 'cargo':
            match = re.search(r'(\d+) passed.*?(\d+) failed', output)
            if match:
                passed_count = int(match.group(1))
                failed_count = int(match.group(2))
                return failed_count == 0, passed_count + failed_count, failed_count, errors
        
        # Fallback: check return code
        return returncode == 0, 0, 0 if returncode == 0 else 1, errors
    
    def _save_history(self, passed: bool, total: int, failures: int):
        """Save test result to history."""
        history = []
        if self.test_history_file.exists():
            try:
                history = json.loads(self.test_history_file.read_text())
            except Exception:
                history = []
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'total': total,
            'failures': failures
        })
        
        # Keep last 100 entries
        history = history[-100:]
        safe_write_json(self.test_history_file, history)
    
    def get_trend(self) -> dict:
        """Get test trend analysis."""
        if not self.test_history_file.exists():
            return {'trend': 'unknown', 'recent_pass_rate': 0}
        
        try:
            history = json.loads(self.test_history_file.read_text())
            if len(history) < 3:
                return {'trend': 'unknown', 'recent_pass_rate': 0}
            
            recent = history[-10:]
            pass_rate = sum(1 for h in recent if h['passed']) / len(recent)
            
            older = history[-20:-10] if len(history) >= 20 else history[:len(history)//2]
            old_rate = sum(1 for h in older if h['passed']) / len(older) if older else 0
            
            if pass_rate > old_rate + 0.1:
                trend = 'improving'
            elif pass_rate < old_rate - 0.1:
                trend = 'degrading'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'recent_pass_rate': pass_rate,
                'old_pass_rate': old_rate,
                'total_runs': len(history)
            }
        except Exception:
            return {'trend': 'unknown', 'recent_pass_rate': 0}
    
    def create_fix_tasks(self, test_result: dict, prd_file: Path) -> int:
        """Create FIX tasks for test failures with STRICT deduplication.
        
        CRITICAL: Don't create duplicate FIX tasks for same error!
        Check ALL existing tasks (including completed ones) to prevent loops.
        
        ALSO: Don't create FIX tasks for infrastructure errors (missing tools, etc.)
        """
        if test_result['passed'] or not test_result['errors']:
            return 0
        
        # Filter out infrastructure errors - these are not code bugs
        INFRA_PATTERNS = [
            'no such file or directory',
            'command not found',
            'not found in path',
            'errno 2',
            'container not running',
            'docker exec failed',
            'timeout',
            'connection refused',
            'permission denied',
        ]
        
        def is_infra_error(err: str) -> bool:
            err_lower = err.lower()
            return any(p in err_lower for p in INFRA_PATTERNS)
        
        # Remove infrastructure errors
        code_errors = [e for e in test_result['errors'] if not is_infra_error(e)]
        if not code_errors:
            return 0  # Only infra errors, nothing to fix in code
        
        try:
            prd = json.loads(prd_file.read_text())
        except Exception:
            return 0
        
        stories = prd.get('userStories', [])
        
        # Count existing FIX tasks (both pending and done) - limit total
        existing_fix_count = sum(1 for s in stories if s.get('type') == 'fix')
        if existing_fix_count >= 10:
            # Too many FIX tasks already - likely a systemic issue
            # Don't create more, let architect handle it
            return 0
        
        created = 0
        
        # Create FIX task only for UNIQUE code errors
        for error in code_errors[:3]:
            # Normalize error for comparison
            error_key = self._normalize_error(error)
            
            # Check ALL existing tasks (including completed!) for similar error
            is_duplicate = False
            for s in stories:
                if s.get('type') != 'fix':
                    continue
                existing_text = (s.get('title', '') + ' ' + s.get('description', '')).lower()
                existing_key = self._normalize_error(existing_text)
                
                # Check if same error pattern
                if error_key == existing_key:
                    is_duplicate = True
                    # If completed FIX has same error, mark it as NOT completed
                    # (the fix didn't work)
                    if s.get('passes'):
                        s['passes'] = False
                        s['description'] += '\n\n[Previous fix attempt did not resolve the issue]'
                    break
            
            if is_duplicate:
                continue
            
            # Generate stable ID based on error content (not timestamp)
            # SECURITY FIX: Use SHA256 instead of MD5 for better collision resistance
            # Use 12 chars instead of 8 for 48 bits of entropy
            error_hash = hashlib.sha256(error_key.encode()).hexdigest()[:12]
            task_id = f"FIX-{error_hash}"
            
            # Check for hash collision by comparing full error_key
            for s in stories:
                if s.get('id') == task_id:
                    # Collision detected - same ID but different error
                    if s.get('_error_key') != error_key:
                        # Append counter to make unique
                        counter = 1
                        while any(st.get('id') == f"{task_id}-{counter}" for st in stories):
                            counter += 1
                        task_id = f"{task_id}-{counter}"
                    break
            
            # Check if task with this ID already exists (after collision handling)
            if any(s.get('id') == task_id for s in stories):
                continue
            
            prd.setdefault('userStories', []).append({
                'id': task_id,
                '_error_key': error_key,  # Store for collision detection
                'title': f'Fix: {error[:80]}',
                'description': f'Test failure detected. Error: {error[:500]}',
                'acceptance': 'All tests pass',
                'passes': False,
                'priority': 0,
                'type': 'fix'
            })
            created += 1
        
        if created > 0 or any(s.get('type') == 'fix' and not s.get('passes') for s in stories):
            prd['verified'] = False
            safe_write_json(prd_file, prd)
        
        return created
    
    def _normalize_error(self, error: str) -> str:
        """Normalize error message for comparison.
        
        Extracts key patterns: file paths, command names, error types.
        Ignores timestamps, line numbers, variable parts.
        """
        error = error.lower()
        
        # Extract key parts
        patterns = []
        
        # Command not found
        if 'command not found' in error or 'not found' in error:
            # Extract what's not found
            match = re.search(r"['\"]?(\w+)['\"]?\s*(?:command\s+)?not\s+found", error)
            if match:
                patterns.append(f"not_found:{match.group(1)}")
        
        # No such file or directory
        if 'no such file' in error:
            match = re.search(r"no such file[^:]*:\s*['\"]?([^'\"\\n]+)", error)
            if match:
                path = match.group(1).strip()
                # Normalize path - keep last 2 components
                parts = path.replace('\\', '/').split('/')
                patterns.append(f"no_file:{'/'.join(parts[-2:])}")
        
        # Permission denied
        if 'permission denied' in error:
            patterns.append("permission_denied")
        
        # Import/module errors
        if 'import' in error or 'module' in error:
            match = re.search(r"(?:import|module)\s+['\"]?(\w+)", error)
            if match:
                patterns.append(f"import:{match.group(1)}")
        
        # If no specific pattern found, use first 50 chars (normalized)
        if not patterns:
            clean = re.sub(r'[^a-z0-9\s]', '', error[:50])
            patterns.append(clean)
        
        return '|'.join(sorted(patterns))


# =============================================================================
# STUCK DETECTION & RECOVERY - Detect loops and auto-recover
# =============================================================================

class StuckDetector:
    """
    Detects when Ralph is stuck and triggers recovery.
    
    Stuck conditions:
    - Same task attempted 3+ times
    - Same error repeated 5+ times
    - No progress for 10+ iterations
    - Verification failed 3+ times in a row
    """
    
    def __init__(self, ralph_dir: Path):
        self.ralph_dir = ralph_dir
        self.stuck_file = ralph_dir / "stuck_history.json"
        self.max_task_attempts = 3
        self.max_same_errors = 5
        self.max_no_progress = 10
        self.max_verification_fails = 3
    
    def check_stuck(self, current_task_id: str, iteration: int) -> Tuple[bool, str]:
        """
        Check if we're stuck.
        
        Returns:
            (is_stuck, reason)
        """
        history = self._load_history()
        
        # Check task attempts
        task_attempts = [h for h in history if h.get('task_id') == current_task_id]
        if len(task_attempts) >= self.max_task_attempts:
            return True, f"Task {current_task_id} attempted {len(task_attempts)} times"
        
        # Check same error
        recent_errors = [h.get('error', '') for h in history[-self.max_same_errors:] if h.get('error')]
        if len(recent_errors) >= self.max_same_errors and len(set(recent_errors)) == 1:
            return True, f"Same error repeated {len(recent_errors)} times: {recent_errors[0][:50]}"
        
        # Check no progress
        recent = history[-self.max_no_progress:]
        if len(recent) >= self.max_no_progress:
            completed = sum(1 for h in recent if h.get('completed'))
            if completed == 0:
                return True, f"No task completed in last {self.max_no_progress} iterations"
        
        # Check verification failures
        recent_verifications = [h for h in history[-10:] if h.get('type') == 'verification']
        failed_verifications = [v for v in recent_verifications if not v.get('passed')]
        if len(failed_verifications) >= self.max_verification_fails:
            return True, f"Verification failed {len(failed_verifications)} times in a row"
        
        return False, ""
    
    def record_attempt(self, task_id: str, iteration: int, completed: bool, 
                       error: Optional[str] = None, attempt_type: str = 'worker'):
        """Record a task attempt."""
        history = self._load_history()
        history.append({
            'task_id': task_id,
            'iteration': iteration,
            'completed': completed,
            'error': error[:200] if error else None,
            'type': attempt_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep last 100 entries
        history = history[-100:]
        self._save_history(history)
    
    def get_recovery_strategy(self, reason: str) -> str:
        """
        Determine recovery strategy based on stuck reason.
        
        Returns one of:
        - 'skip_task': Skip current task, mark as blocked
        - 'rollback': Rollback recent changes
        - 'split_task': Try to split task into smaller pieces
        - 'escalate': Add architect review task
        """
        if 'attempted' in reason:
            return 'skip_task'
        elif 'error repeated' in reason:
            return 'rollback'
        elif 'No task completed' in reason:
            return 'escalate'
        elif 'Verification failed' in reason:
            return 'rollback'
        else:
            return 'skip_task'
    
    def _load_history(self) -> List[dict]:
        """Load stuck history."""
        if self.stuck_file.exists():
            try:
                return json.loads(self.stuck_file.read_text())
            except Exception:
                return []
        return []
    
    def _save_history(self, history: List[dict]):
        """Save stuck history."""
        safe_write_json(self.stuck_file, history)
    
    def clear_history(self):
        """Clear stuck history (call after successful recovery)."""
        self._save_history([])
    
    def get_task_attempts(self, task_id: str) -> str:
        """Get formatted history of previous attempts on a task."""
        history = self._load_history()
        task_attempts = [h for h in history if h.get('task_id') == task_id and not h.get('completed')]
        
        if not task_attempts:
            return "First attempt at this task."
        
        lines = [f"⚠️ **{len(task_attempts)} FAILED attempt(s) - try DIFFERENT approach!**\n"]
        for i, attempt in enumerate(task_attempts[-3:], 1):  # Last 3 failed attempts
            error = attempt.get('error', 'Unknown error')
            iteration = attempt.get('iteration', '?')
            lines.append(f"- Attempt {i} (iter {iteration}): {error[:100]}")
        
        return "\n".join(lines)
    
    def _normalize_error(self, error: str) -> set:
        """Normalize error text for better matching.
        
        Extracts key terms: error types, file names, function names.
        Removes noise: line numbers, timestamps, paths.
        """
        import re
        text = error.lower()
        
        # Extract key error patterns
        keywords = set()
        
        # Error types (e.g., TypeError, ModuleNotFoundError, cargo, npm)
        error_types = re.findall(r'\b(error|failed|not found|undefined|null|missing|invalid|cannot|unable)\b', text)
        keywords.update(error_types)
        
        # Tool/command names
        tools = re.findall(r'\b(cargo|npm|pip|python|node|rust|go|java|make|cmake|gcc|clang)\b', text)
        keywords.update(tools)
        
        # Module/package names (words before "not found" or after "import")
        modules = re.findall(r"(?:import|from|require|use)\s+['\"]?(\w+)", text)
        modules += re.findall(r"(\w+)(?:\s+not found|:\s+not found)", text)
        keywords.update(modules)
        
        # File extensions as hints
        extensions = re.findall(r'\.(py|rs|js|ts|go|java|cpp|c|h)\b', text)
        keywords.update(extensions)
        
        # Also include significant words (3+ chars, not common)
        stopwords = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'line', 'file', 'error'}
        words = [w for w in re.findall(r'\b[a-z]{3,}\b', text) if w not in stopwords]
        keywords.update(words[:15])  # Top 15 significant words
        
        return keywords
    
    def get_fix_history(self, error_desc: str) -> str:
        """Get history of ALL FIX attempts for similar errors.
        
        For FIX tasks: shows what was tried before for this type of error,
        even across different FIX task IDs.
        Uses normalized error matching for better recall.
        """
        history = self._load_history()
        
        # Normalize current error
        error_keywords = self._normalize_error(error_desc)
        similar_attempts = []
        
        for h in history:
            task_id = h.get('task_id', '')
            if not task_id.startswith('FIX'):
                continue
            attempt_error = h.get('error', '')
            attempt_keywords = self._normalize_error(attempt_error)
            
            # Check overlap - at least 3 common keywords OR 30% overlap
            overlap = len(error_keywords & attempt_keywords)
            min_size = min(len(error_keywords), len(attempt_keywords)) or 1
            if overlap >= 3 or (overlap / min_size) >= 0.3:
                similar_attempts.append(h)
        
        if not similar_attempts:
            return ""
        
        lines = [f"🔧 **{len(similar_attempts)} previous FIX attempt(s) for similar error:**\n"]
        for attempt in similar_attempts[-5:]:  # Last 5
            task_id = attempt.get('task_id', '?')
            iteration = attempt.get('iteration', '?')
            error = attempt.get('error', 'Unknown')[:60]
            completed = "✓" if attempt.get('completed') else "✗"
            lines.append(f"- [{completed}] {task_id} (iter {iteration}): {error}")
        
        lines.append("\n**Learn from these - try something DIFFERENT!**")
        return "\n".join(lines)


# =============================================================================
# HIERARCHICAL MEMORY - Hot/Warm/Cold context management
# =============================================================================

class HierarchicalMemory:
    """
    Three-tier memory system for optimal context usage.
    
    - Hot: Last 5 iterations, full detail
    - Warm: Last 20 iterations, summaries
    - Cold: Key points only, permanent storage
    """
    
    def __init__(self, ralph_dir: Path):
        self.ralph_dir = ralph_dir
        self.memory_dir = ralph_dir / "memory"
        # Directory is created by RalphManager via Docker
        
        self.hot_file = self.memory_dir / "hot.json"
        self.warm_file = self.memory_dir / "warm.json"
        self.cold_file = self.memory_dir / "cold.json"
        
        # Optimized for 200K context models - more memory = smarter
        self.hot_limit = 10     # Last 10 iterations with full details
        self.warm_limit = 40    # 11-50 iterations with summaries
        self.cold_limit = 200   # Key points from all history
    
    def add_iteration(self, iteration: int, task_id: str, summary: str, 
                      details: str, key_points: List[str] = None):
        """Add an iteration to memory."""
        entry = {
            'iteration': iteration,
            'task_id': task_id,
            'summary': summary,
            'details': details,
            'key_points': key_points or [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to hot
        hot = self._load(self.hot_file)
        hot.append(entry)
        
        # Promote old hot to warm
        while len(hot) > self.hot_limit:
            old = hot.pop(0)
            self._add_to_warm(old)
        
        self._save(self.hot_file, hot)
    
    def _add_to_warm(self, entry: dict):
        """Add summarized entry to warm storage."""
        warm = self._load(self.warm_file)
        
        # Summarize: keep only summary and key points
        warm_entry = {
            'iteration': entry['iteration'],
            'task_id': entry['task_id'],
            'summary': entry['summary'][:500],
            'key_points': entry.get('key_points', [])[:5],
            'timestamp': entry['timestamp']
        }
        warm.append(warm_entry)
        
        # Promote old warm to cold
        while len(warm) > self.warm_limit:
            old = warm.pop(0)
            self._add_to_cold(old)
        
        self._save(self.warm_file, warm)
    
    def _add_to_cold(self, entry: dict):
        """Add key points to cold storage."""
        cold = self._load(self.cold_file)
        
        # Keep only key points
        for point in entry.get('key_points', []):
            if point and point not in cold:
                cold.append(point)
        
        # Deduplicate and limit
        cold = list(dict.fromkeys(cold))[-self.cold_limit:]
        self._save(self.cold_file, cold)
    
    def get_context(self, task_keywords: List[str] = None, max_chars: int = 30000) -> str:
        """
        Get memory context for prompt.
        
        Args:
            task_keywords: Keywords to prioritize relevant content
            max_chars: Maximum characters to return
        """
        parts = []
        remaining = max_chars
        
        # Always include hot (full detail)
        hot = self._load(self.hot_file)
        if hot:
            hot_text = "## Recent Iterations (detailed)\n"
            for entry in hot[-3:]:  # Last 3
                hot_text += f"\n### Iteration {entry['iteration']}: {entry['task_id']}\n"
                hot_text += entry.get('details', entry.get('summary', ''))[:1000]
                hot_text += "\n"
            parts.append(hot_text)
            remaining -= len(hot_text)
        
        # Add relevant warm
        if remaining > 5000:
            warm = self._load(self.warm_file)
            if warm and task_keywords:
                # Score by relevance
                scored = []
                for entry in warm:
                    text = entry.get('summary', '') + ' '.join(entry.get('key_points', []))
                    score = sum(1 for kw in task_keywords if kw.lower() in text.lower())
                    scored.append((score, entry))
                
                # Take top 5 by relevance
                scored.sort(reverse=True, key=lambda x: x[0])
                relevant = [e for s, e in scored[:5] if s > 0]
                
                if relevant:
                    warm_text = "\n## Related Past Work\n"
                    for entry in relevant:
                        warm_text += f"- [{entry['task_id']}] {entry['summary'][:200]}\n"
                    parts.append(warm_text)
                    remaining -= len(warm_text)
        
        # Add cold (key points)
        if remaining > 2000:
            cold = self._load(self.cold_file)
            if cold:
                # Filter by relevance if keywords provided
                if task_keywords:
                    cold = [p for p in cold if any(kw.lower() in p.lower() for kw in task_keywords)]
                
                if cold:
                    cold_text = "\n## Key Learnings\n"
                    cold_text += '\n'.join(f"- {p}" for p in cold[-20:])
                    parts.append(cold_text)
        
        return '\n'.join(parts)
    
    def add_permanent(self, point: str):
        """Add a point directly to cold/permanent storage."""
        cold = self._load(self.cold_file)
        if point not in cold:
            cold.append(point)
            cold = cold[-self.cold_limit:]
            self._save(self.cold_file, cold)
    
    def _load(self, filepath: Path) -> List:
        """Load JSON file."""
        if filepath.exists():
            try:
                return json.loads(filepath.read_text())
            except Exception:
                return []
        return []
    
    def _save(self, filepath: Path, data: List):
        """Save JSON file."""
        safe_write_json(filepath, data)


# =============================================================================
# SEMANTIC SEARCH - Sentence Transformers based relevance scoring
# Uses all-mpnet-base-v2 model for true semantic similarity
# =============================================================================

# Maximum entries for semantic cache to prevent memory leaks
MAX_SEMANTIC_CACHE = 10000

class SemanticSearch:
    """
    Semantic search using Sentence Transformers embeddings.
    
    Provides true semantic similarity search for learnings, memory, and deduplication.
    Uses all-mpnet-base-v2 model - best quality for English text.
    
    MEMORY FIX: Uses LRU eviction to cap cache at MAX_SEMANTIC_CACHE entries
    to prevent unbounded memory growth during long sessions.
    """
    
    # Class-level model cache (shared across instances)
    _model = None
    _model_loaded = False
    
    def __init__(self, ralph_dir: Path):
        self.ralph_dir = ralph_dir
        self.cache_file = ralph_dir / "semantic_cache.json"
        self._embeddings_cache = {}  # text_hash -> embedding
        # LRU cache: track access order for eviction (oldest first)
        self._cache_access_order: deque = deque(maxlen=MAX_SEMANTIC_CACHE)
        self._load_cache()
    
    @classmethod
    def _get_model(cls):
        """Lazy-load Sentence Transformer model (once per process)."""
        if cls._model_loaded:
            return cls._model
        
        cls._model_loaded = True
        from sentence_transformers import SentenceTransformer
        cls._model = SentenceTransformer('all-mpnet-base-v2')
        return cls._model
    
    def _load_cache(self):
        """Load embeddings cache from disk with LRU tracking."""
        if self.cache_file.exists():
            try:
                import json
                with open(self.cache_file, 'r') as f:
                    self._embeddings_cache = json.load(f)
                # Initialize access order with existing keys (limited to max)
                keys = list(self._embeddings_cache.keys())[-MAX_SEMANTIC_CACHE:]
                self._cache_access_order = deque(keys, maxlen=MAX_SEMANTIC_CACHE)
            except Exception:
                self._embeddings_cache = {}
                self._cache_access_order = deque(maxlen=MAX_SEMANTIC_CACHE)
    
    def _save_cache(self):
        """Save embeddings cache to disk (LRU entries only)."""
        try:
            # SECURITY FIX: Only save entries in LRU order to prevent unbounded growth
            to_save = {k: self._embeddings_cache[k] 
                      for k in self._cache_access_order 
                      if k in self._embeddings_cache}
            with open(self.cache_file, 'w') as f:
                json.dump(to_save, f)
        except Exception:
            pass
    
    def _text_hash(self, text: str) -> str:
        """Create hash for text to use as cache key."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _get_embedding(self, text: str):
        """Get embedding for text (with LRU cache).
        
        MEMORY FIX: Implements LRU eviction to cap cache at MAX_SEMANTIC_CACHE
        entries. Each 768-dim float32 embedding is ~3KB, so 10k entries = ~30MB.
        """
        import numpy as np
        model = self._get_model()
        
        text_hash = self._text_hash(text)
        
        if text_hash in self._embeddings_cache:
            # LRU FIX: Move to end (most recently used)
            if text_hash in self._cache_access_order:
                self._cache_access_order.remove(text_hash)
            self._cache_access_order.append(text_hash)
            return np.array(self._embeddings_cache[text_hash], dtype=np.float32)
        
        # LRU FIX: Evict oldest if at limit BEFORE adding new
        while len(self._embeddings_cache) >= MAX_SEMANTIC_CACHE:
            if self._cache_access_order:
                oldest = self._cache_access_order.popleft()
                self._embeddings_cache.pop(oldest, None)
            else:
                break
        
        embedding = model.encode(text, convert_to_numpy=True)
        self._embeddings_cache[text_hash] = embedding.tolist()
        self._cache_access_order.append(text_hash)
        return embedding.astype(np.float32)
    
    def _get_embeddings_batch(self, texts: List[str]):
        """Get embeddings for multiple texts efficiently (with LRU cache)."""
        import numpy as np
        model = self._get_model()
        
        results = []
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            if text_hash in self._embeddings_cache:
                # LRU FIX: Update access order for cache hit
                if text_hash in self._cache_access_order:
                    self._cache_access_order.remove(text_hash)
                self._cache_access_order.append(text_hash)
                results.append((i, np.array(self._embeddings_cache[text_hash], dtype=np.float32)))
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        if texts_to_encode:
            # LRU FIX: Make room for new entries
            for text in texts_to_encode:
                while len(self._embeddings_cache) >= MAX_SEMANTIC_CACHE:
                    if self._cache_access_order:
                        oldest = self._cache_access_order.popleft()
                        self._embeddings_cache.pop(oldest, None)
                    else:
                        break
            
            new_embeddings = model.encode(texts_to_encode, convert_to_numpy=True)
            for idx, text, emb in zip(indices_to_encode, texts_to_encode, new_embeddings):
                text_hash = self._text_hash(text)
                self._embeddings_cache[text_hash] = emb.tolist()
                self._cache_access_order.append(text_hash)
                results.append((idx, emb.astype(np.float32)))
        
        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results], dtype=np.float32)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        from sentence_transformers.util import cos_sim
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        return float(cos_sim(emb1, emb2)[0][0])
    
    def find_similar(self, query: str, documents: List[str], top_k: int = 5, 
                     threshold: float = 0.1) -> List[Tuple[int, float, str]]:
        """
        Find most similar documents to query using semantic similarity.
        
        Returns: List of (index, score, document) tuples sorted by similarity.
        """
        if not documents:
            return []
        
        from sentence_transformers.util import cos_sim
        
        query_emb = self._get_embedding(query)
        doc_embs = self._get_embeddings_batch(documents)
        
        similarities = cos_sim(query_emb, doc_embs)[0].numpy()
        
        results = []
        for i, (score, doc) in enumerate(zip(similarities, documents)):
            if score >= threshold:
                results.append((i, float(score), doc))
        
        results.sort(key=lambda x: x[1], reverse=True)
        self._save_cache()
        return results[:top_k]
    
    def is_duplicate(self, new_text: str, existing_texts: List[str], 
                     threshold: float = 0.75) -> bool:
        """Check if new_text is semantically duplicate of any existing text."""
        if not existing_texts:
            return False
        
        similar = self.find_similar(new_text, existing_texts, top_k=1, threshold=threshold)
        return len(similar) > 0
    
    def select_relevant(self, query: str, sections: List[str], 
                        max_chars: int = 15000) -> str:
        """Select most relevant sections for query within char budget."""
        if not sections:
            return ""
        
        total_size = sum(len(s) for s in sections)
        if total_size <= max_chars:
            return '\n\n'.join(sections)
        
        similar = self.find_similar(query, sections, top_k=len(sections), threshold=0.15)
        
        # Always include last 2 sections (most recent)
        recent_indices = set(range(max(0, len(sections) - 2), len(sections)))
        
        result = []
        used_indices = set()
        remaining = max_chars
        
        for idx in recent_indices:
            section = sections[idx]
            if len(section) <= remaining:
                result.append((idx, section))
                used_indices.add(idx)
                remaining -= len(section)
        
        for idx, score, section in similar:
            if idx in used_indices:
                continue
            if len(section) <= remaining:
                result.append((idx, section))
                used_indices.add(idx)
                remaining -= len(section)
        
        result.sort(key=lambda x: x[0])
        return '\n\n'.join(s for _, s in result)


# =============================================================================
# QMD SEMANTIC CODE SEARCH - Optional integration for large codebases
# =============================================================================

class QMDManager:
    """
    Manage QMD (Query Markup Documents) semantic search for a project.
    
    QMD provides hybrid search (BM25 + vector + LLM reranking) for code/docs.
    Useful for large existing codebases where grep/find isn't enough.
    
    Installation happens inside Docker container via MCP.
    """
    
    # QMD MCP config template
    MCP_CONFIG = {
        "command": "qmd",
        "args": ["mcp"]
    }
    
    # Setup script to run inside container
    SETUP_SCRIPT = '''#!/bin/bash
set -e

# Install bun if not present
if ! command -v bun &> /dev/null; then
    echo "[QMD] Installing bun..."
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="$HOME/.bun"
    export PATH="$BUN_INSTALL/bin:$PATH"
fi

# Install/update qmd
echo "[QMD] Installing qmd..."
bun install -g https://github.com/tobi/qmd

# Create collection for workspace
echo "[QMD] Creating collection..."
qmd collection add /workspace --name code 2>/dev/null || true

# Generate embeddings (this downloads models on first run)
echo "[QMD] Generating embeddings (first run downloads ~2GB models)..."
qmd embed

echo "[QMD] Setup complete!"
'''
    
    # Reindex script (for periodic updates)
    REINDEX_SCRIPT = '''#!/bin/bash
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

if command -v qmd &> /dev/null; then
    qmd embed --quiet
fi
'''
    
    def __init__(self, project: 'Project'):
        self.project = project
        self.config_file = project.workspace / ".qmd_enabled"
    
    @property
    def is_enabled(self) -> bool:
        """Check if QMD is enabled for this project."""
        return self.config_file.exists()
    
    def enable(self) -> Tuple[bool, str]:
        """Enable QMD for this project."""
        try:
            # Create marker file
            self.config_file.write_text(datetime.now().isoformat())
            
            # Add to MCP config
            self._update_mcp_config(add=True)
            
            # Create setup script in workspace
            setup_script = self.project.workspace / ".qmd_setup.sh"
            setup_script.write_text(self.SETUP_SCRIPT)
            setup_script.chmod(0o755)
            
            # Create reindex script
            reindex_script = self.project.workspace / ".qmd_reindex.sh"
            reindex_script.write_text(self.REINDEX_SCRIPT)
            reindex_script.chmod(0o755)
            
            return True, "QMD enabled. Run setup inside container."
        except Exception as e:
            return False, f"Failed to enable QMD: {e}"
    
    def disable(self) -> Tuple[bool, str]:
        """Disable QMD for this project."""
        try:
            if self.config_file.exists():
                self.config_file.unlink()
            
            # Remove from MCP config
            self._update_mcp_config(add=False)
            
            # Remove scripts
            for script in [".qmd_setup.sh", ".qmd_reindex.sh"]:
                script_path = self.project.workspace / script
                if script_path.exists():
                    script_path.unlink()
            
            return True, "QMD disabled"
        except Exception as e:
            return False, f"Failed to disable QMD: {e}"
    
    def _update_mcp_config(self, add: bool):
        """Add or remove QMD from MCP servers config."""
        mcp_file = self.project.workspace / ".openhands" / "mcp_servers.json"
        
        if not mcp_file.exists():
            if add:
                mcp_file.parent.mkdir(parents=True, exist_ok=True)
                config = {"mcpServers": {"qmd": self.MCP_CONFIG}}
                mcp_file.write_text(json.dumps(config, indent=2))
            return
        
        try:
            config = json.loads(mcp_file.read_text())
            if "mcpServers" not in config:
                config["mcpServers"] = {}
            
            if add:
                config["mcpServers"]["qmd"] = self.MCP_CONFIG
            else:
                config["mcpServers"].pop("qmd", None)
            
            mcp_file.write_text(json.dumps(config, indent=2))
        except Exception:
            pass
    
    def get_setup_command(self) -> str:
        """Get command to run inside container to setup QMD."""
        return "bash /workspace/.qmd_setup.sh"
    
    def get_reindex_command(self) -> str:
        """Get command to reindex files."""
        return "bash /workspace/.qmd_reindex.sh"


# =============================================================================
# METRICS TRACKING
# =============================================================================

class RalphMetrics:
    """
    Track Ralph performance metrics for monitoring and optimization.
    
    Metrics tracked:
    - Iteration success/failure rate
    - Time per iteration
    - Context size growth
    - Stuck frequency
    - Condense effectiveness
    """
    
    def __init__(self, ralph_dir: Path):
        self.ralph_dir = ralph_dir
        self.metrics_file = ralph_dir / "metrics.json"
        self._metrics = self._load()
    
    def _load(self) -> dict:
        """Load metrics from file."""
        if self.metrics_file.exists():
            try:
                return json.loads(self.metrics_file.read_text())
            except Exception:
                pass
        return {
            "total_iterations": 0,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "stuck_count": 0,
            "condense_count": 0,
            "total_time_seconds": 0,
            "iteration_times": [],  # Last 50 iteration times
            "context_sizes": [],    # Last 50 context sizes (tokens)
            "learnings_count": 0,
            "learnings_dedupe_skipped": 0,
            "condense_facts_preserved": 0,
            "condense_facts_lost": 0,
            "started_at": None,
            "last_updated": None
        }
    
    def _save(self):
        """Save metrics to file."""
        self._metrics["last_updated"] = datetime.now().isoformat()
        try:
            safe_write_json(self.metrics_file, self._metrics)
        except Exception as e:
            log_error(f"Failed to save metrics: {e}")
    
    def record_iteration(self, success: bool, time_seconds: float, 
                         context_tokens: int, stuck: bool = False):
        """Record metrics for one iteration."""
        if self._metrics["started_at"] is None:
            self._metrics["started_at"] = datetime.now().isoformat()
        
        self._metrics["total_iterations"] += 1
        if success:
            self._metrics["successful_iterations"] += 1
        else:
            self._metrics["failed_iterations"] += 1
        
        if stuck:
            self._metrics["stuck_count"] += 1
        
        self._metrics["total_time_seconds"] += time_seconds
        
        # Keep last 50 iteration times
        self._metrics["iteration_times"].append(time_seconds)
        self._metrics["iteration_times"] = self._metrics["iteration_times"][-50:]
        
        # Keep last 50 context sizes
        self._metrics["context_sizes"].append(context_tokens)
        self._metrics["context_sizes"] = self._metrics["context_sizes"][-50:]
        
        self._save()
    
    def record_condense(self, facts_preserved: int, facts_total: int):
        """Record condense operation metrics."""
        self._metrics["condense_count"] += 1
        self._metrics["condense_facts_preserved"] += facts_preserved
        self._metrics["condense_facts_lost"] += (facts_total - facts_preserved)
        self._save()
    
    def record_learning(self, added: bool, was_duplicate: bool = False):
        """Record learning addition."""
        if added:
            self._metrics["learnings_count"] += 1
        if was_duplicate:
            self._metrics["learnings_dedupe_skipped"] += 1
        self._save()
    
    def get_stats(self) -> dict:
        """Get computed statistics."""
        m = self._metrics
        total = m["total_iterations"]
        
        stats = {
            "total_iterations": total,
            "success_rate": m["successful_iterations"] / total if total > 0 else 0,
            "stuck_rate": m["stuck_count"] / total if total > 0 else 0,
            "avg_time_seconds": sum(m["iteration_times"]) / len(m["iteration_times"]) if m["iteration_times"] else 0,
            "avg_context_tokens": sum(m["context_sizes"]) / len(m["context_sizes"]) if m["context_sizes"] else 0,
            "context_growth_rate": self._compute_growth_rate(m["context_sizes"]),
            "learnings_total": m["learnings_count"],
            "learnings_dedupe_rate": m["learnings_dedupe_skipped"] / (m["learnings_count"] + m["learnings_dedupe_skipped"]) if (m["learnings_count"] + m["learnings_dedupe_skipped"]) > 0 else 0,
            "condense_preservation_rate": m["condense_facts_preserved"] / (m["condense_facts_preserved"] + m["condense_facts_lost"]) if (m["condense_facts_preserved"] + m["condense_facts_lost"]) > 0 else 1.0
        }
        return stats
    
    def _compute_growth_rate(self, sizes: List[int]) -> float:
        """Compute context size growth rate."""
        if len(sizes) < 10:
            return 0.0
        # Compare first half avg to second half avg
        mid = len(sizes) // 2
        first_half = sum(sizes[:mid]) / mid
        second_half = sum(sizes[mid:]) / (len(sizes) - mid)
        if first_half == 0:
            return 0.0
        return (second_half - first_half) / first_half
    
    def record_task_result(self, task_id: str, success: bool, duration_sec: float, error: str = ""):
        """Record task completion result for success metrics by type."""
        if "task_metrics" not in self._metrics:
            self._metrics["task_metrics"] = {
                "by_type": {},  # TASK, FIX, CLEAN etc
                "recent_failures": []  # Last 20 failures with reasons
            }
        
        # Determine task type from ID prefix
        task_type = "TASK"
        for prefix in ["FIX", "CLEAN", "REFACTOR", "TEST", "DOC"]:
            if task_id.startswith(prefix):
                task_type = prefix
                break
        
        # Update by-type stats
        if task_type not in self._metrics["task_metrics"]["by_type"]:
            self._metrics["task_metrics"]["by_type"][task_type] = {
                "total": 0, "success": 0, "total_time": 0
            }
        
        type_stats = self._metrics["task_metrics"]["by_type"][task_type]
        type_stats["total"] += 1
        type_stats["total_time"] += duration_sec
        if success:
            type_stats["success"] += 1
        else:
            # Track failure reasons
            self._metrics["task_metrics"]["recent_failures"].append({
                "task_id": task_id,
                "type": task_type,
                "error": error[:100] if error else "unknown",
                "time": datetime.now().isoformat()
            })
            # Keep last 20
            self._metrics["task_metrics"]["recent_failures"] = \
                self._metrics["task_metrics"]["recent_failures"][-20:]
        
        self._save()
    
    def get_task_stats(self) -> dict:
        """Get task success statistics by type."""
        if "task_metrics" not in self._metrics:
            return {}
        
        result = {}
        for task_type, stats in self._metrics["task_metrics"]["by_type"].items():
            total = stats["total"]
            success = stats["success"]
            result[task_type] = {
                "total": total,
                "success": success,
                "rate": success / total if total > 0 else 0,
                "avg_time": stats["total_time"] / total if total > 0 else 0
            }
        return result
    
    def get_common_failures(self) -> List[str]:
        """Get most common failure reasons."""
        if "task_metrics" not in self._metrics:
            return []
        
        failures = self._metrics["task_metrics"].get("recent_failures", [])
        error_counts = {}
        for f in failures:
            err = f.get("error", "unknown")[:50]
            error_counts[err] = error_counts.get(err, 0) + 1
        
        return sorted(error_counts.keys(), key=lambda x: error_counts[x], reverse=True)[:5]
    
    def format_summary(self) -> str:
        """Format metrics as human-readable summary."""
        stats = self.get_stats()
        task_stats = self.get_task_stats()
        
        # Build task metrics section
        task_lines = []
        for task_type, ts in sorted(task_stats.items(), key=lambda x: x[1]["total"], reverse=True):
            rate_pct = ts["rate"] * 100
            task_lines.append(f"  {task_type}: {rate_pct:.0f}% ({ts['success']}/{ts['total']}), avg {ts['avg_time']:.0f}s")
        
        task_section = "\n".join(task_lines) if task_lines else "  (no task data yet)"
        
        # Common failures
        failures = self.get_common_failures()
        fail_section = "\n".join(f"  - {f}" for f in failures[:3]) if failures else "  (none)"
        
        return f"""## Ralph Metrics Summary

**Iterations:** {stats['total_iterations']} total ({stats['success_rate']*100:.1f}% success)
**Stuck Rate:** {stats['stuck_rate']*100:.1f}%
**Avg Time:** {stats['avg_time_seconds']:.1f}s per iteration
**Context:** {stats['avg_context_tokens']:.0f} tokens avg, {stats['context_growth_rate']*100:+.1f}% growth

**Task Success by Type:**
{task_section}

**Common Failures:**
{fail_section}

**Learnings:** {stats['learnings_total']} total, {stats['learnings_dedupe_rate']*100:.1f}% duplicates skipped
**Condense:** {stats['condense_preservation_rate']*100:.1f}% facts preserved
"""


# =============================================================================
# LEARNINGS MANAGER WITH DEDUPLICATION
# =============================================================================

class LearningsManager:
    """
    Manage learnings with semantic deduplication.
    
    Prevents duplicate information from accumulating.
    """
    
    def __init__(self, ralph_dir: Path, semantic_search: SemanticSearch):
        self.ralph_dir = ralph_dir
        self.learnings_file = ralph_dir / "learnings.md"
        self.semantic = semantic_search
        self._entries = []
        self._load()
    
    def _load(self):
        """Load and parse existing learnings."""
        if not self.learnings_file.exists():
            return
        
        content = self.learnings_file.read_text()
        # Split by section headers
        sections = re.split(r'\n(?=###?\s)', content)
        self._entries = [s.strip() for s in sections if s.strip()]
    
    def add(self, learning: str, metrics: Optional['RalphMetrics'] = None) -> bool:
        """
        Add learning if not duplicate.
        
        Returns True if added, False if duplicate.
        """
        learning = learning.strip()
        if not learning:
            return False
        
        # Check for semantic duplicate (threshold 0.55 for sentence-transformers)
        if self._entries:
            if self.semantic.is_duplicate(learning, self._entries, threshold=0.55):
                if metrics:
                    metrics.record_learning(added=False, was_duplicate=True)
                return False
        
        # Add new learning
        self._entries.append(learning)
        self._save()
        
        if metrics:
            metrics.record_learning(added=True, was_duplicate=False)
        
        return True
    
    def _save(self):
        """Save learnings to file."""
        content = '\n\n'.join(self._entries)
        safe_write_text(self.learnings_file, content)
    
    def get_relevant(self, query: str, max_chars: int = 15000) -> str:
        """Get learnings relevant to query."""
        if not self._entries:
            return ""
        return self.semantic.select_relevant(query, self._entries, max_chars)
    
    def get_all(self) -> str:
        """Get all learnings."""
        return '\n\n'.join(self._entries)
    
    def count(self) -> int:
        """Get number of learning entries."""
        return len(self._entries)


# =============================================================================
# CONTEXT CONDENSER - Summarize entire conversation periodically
# =============================================================================

class ContextCondenser:
    """
    Summarizes entire conversation context using the same model.
    
    Creates condensed summaries that preserve:
    - Critical decisions and reasoning
    - Errors and solutions
    - Project state
    - Patterns and anti-patterns
    """
    
    CONDENSE_PROMPT_TEMPLATE = '''# Context Condensation - Iteration {iteration}

You are condensing the context for a long-running autonomous coding session.
Your goal: Create a PERFECT summary that lets future iterations continue seamlessly.

## THE MISSION
{mission}

## RECENT PROGRESS (last {num_iterations} iterations)
{recent_progress}

## CURRENT LEARNINGS
{learnings}

## CURRENT PROJECT STATE
{project_state}

## TASK

Create a comprehensive CONDENSED CONTEXT. This will replace older context.

You MUST preserve:
1. **Mission**: What we're building (keep full mission)
2. **Completed work**: What's done and working
3. **Current state**: Where we are now
4. **All errors**: Every error encountered and how it was fixed
5. **Key decisions**: WHY we chose certain approaches
6. **Patterns**: Code patterns that work in this project
7. **Anti-patterns**: What NOT to do (learned the hard way)
8. **Blockers**: Any unresolved issues
9. **Next steps**: What needs to happen next

Output this EXACT format (will be parsed):

```condensed
## Mission Summary
[1-2 sentence mission summary]

## Project State (Iteration {iteration})
- Phase: [planning/development/verification]
- Tasks: [X/Y completed]
- Last completed: [task name]
- Currently working on: [task name or "none"]

## Completed Milestones
- [Milestone 1]: [brief what was done]
- [Milestone 2]: [brief what was done]

## Critical Decisions
- [Decision]: [Why this was chosen]
- [Decision]: [Why this was chosen]

## Error Log (IMPORTANT - do not repeat these)
- [Error type]: [What happened] → [How fixed]
- [Error type]: [What happened] → [How fixed]

## Code Patterns (use these)
- [Pattern name]: [How to use it]
- [Pattern name]: [How to use it]

## Anti-Patterns (AVOID these)
- [Anti-pattern]: [Why it's bad]
- [Anti-pattern]: [Why it's bad]

## Unresolved Issues
- [Issue]: [Status]

## Build & Run
[Commands to build and run the project]

## Next Actions
1. [Next thing to do]
2. [After that]
```

Be THOROUGH. Missing information = lost forever.
'''
    
    def __init__(self, ralph_dir: Path):
        self.ralph_dir = ralph_dir
        self.condensed_file = ralph_dir / "condensed_context.md"
        self.condense_history_file = ralph_dir / "condense_history.json"
        self.last_condense_iteration = 0
        self._load_state()
    
    def _load_state(self):
        """Load condenser state."""
        if self.condense_history_file.exists():
            try:
                history = json.loads(self.condense_history_file.read_text())
                self.last_condense_iteration = history.get('last_iteration', 0)
            except Exception:
                pass
    
    def _save_state(self, iteration: int):
        """Save condenser state."""
        history = {
            'last_iteration': iteration,
            'timestamp': datetime.now().isoformat()
        }
        safe_write_json(self.condense_history_file, history)
    
    def should_condense(self, iteration: int, config: dict, 
                        context_size: int = 0, next_is_architect: bool = False) -> Tuple[bool, str]:
        """
        Determine if condensation should run.
        
        Returns:
            (should_condense, reason)
        """
        condense_interval = config.get('condenseInterval', 15)
        condense_before_architect = config.get('condenseBeforeArchitect', True)
        
        # Disabled
        if condense_interval <= 0:
            return False, "disabled"
        
        # Not enough iterations since last condense
        iterations_since = iteration - self.last_condense_iteration
        
        # 1. Interval-based
        if iterations_since >= condense_interval:
            return True, f"interval ({iterations_since} iterations)"
        
        # 2. Before architect (if enabled)
        if condense_before_architect and next_is_architect and iterations_since >= 5:
            return True, "before architect review"
        
        # 3. Context too large (emergency)
        if context_size > 150000:  # chars, ~37K tokens
            return True, f"context size ({context_size} chars)"
        
        return False, ""
    
    def build_condense_prompt(self, iteration: int, mission: str, 
                              recent_progress: str, learnings: str, 
                              project_state: str, num_iterations: int = 15) -> str:
        """Build the condensation prompt."""
        return self.CONDENSE_PROMPT_TEMPLATE.format(
            iteration=iteration,
            mission=mission[:10000],  # Limit mission size
            recent_progress=recent_progress[:30000],
            learnings=learnings[:20000],
            project_state=project_state[:10000],
            num_iterations=num_iterations
        )
    
    def parse_condensed_output(self, output: str) -> Optional[str]:
        """Extract condensed context from model output."""
        # Look for ```condensed block
        match = re.search(r'```condensed\s*\n(.*?)\n```', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for ## Mission Summary
        match = re.search(r'(## Mission Summary.*)', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Last resort: return everything after "## "
        if '## ' in output:
            idx = output.index('## ')
            return output[idx:].strip()
        
        return None
    
    def save_condensed(self, iteration: int, condensed: str):
        """Save condensed context."""
        # Add header
        header = f"# Condensed Context (Iteration {iteration})\n"
        header += f"# Generated: {datetime.now().isoformat()}\n"
        header += f"# Previous iterations summarized into this document\n\n"
        
        full_content = header + condensed
        
        # Backup old condensed if exists
        if self.condensed_file.exists():
            backup_dir = self.ralph_dir / "condense_backups"
            # Directory created by RalphManager via Docker
            backup_file = backup_dir / f"condensed_{self.last_condense_iteration}.md"
            try:
                shutil.copy2(self.condensed_file, backup_file)
                # Keep only last 5 backups
                backups = sorted(backup_dir.glob("condensed_*.md"))
                for old_backup in backups[:-5]:
                    old_backup.unlink()
            except Exception:
                pass  # Backup is best-effort
        
        # Save new condensed
        safe_write_text(self.condensed_file, full_content)
        self._save_state(iteration)
        self.last_condense_iteration = iteration
    
    def get_condensed_context(self) -> str:
        """Get current condensed context if exists."""
        if self.condensed_file.exists():
            return self.condensed_file.read_text()
        return ""
    
    def get_recent_iterations_summary(self, iterations_dir: Path, 
                                       num_iterations: int = 15) -> str:
        """Build summary of recent iterations from log files."""
        summaries = []
        
        # Get iteration files
        if not iterations_dir.exists():
            return ""
        
        json_files = sorted(iterations_dir.glob("iteration_*.json"))[-num_iterations:]
        
        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text())
                iter_num = data.get('iteration', '?')
                iter_type = data.get('type', '?')
                result = data.get('result', 'unknown')
                
                # Get key messages
                messages = data.get('assistant_messages', [])
                key_content = ""
                if messages:
                    # Get first and last message snippets
                    first = messages[0][:500] if messages else ""
                    last = messages[-1][:500] if len(messages) > 1 else ""
                    key_content = f"\n  First: {first[:200]}...\n  Last: {last[:200]}..."
                
                summaries.append(f"### Iteration {iter_num} ({iter_type})\n"
                               f"Result: {result}{key_content}")
            except Exception:
                continue
        
        return '\n\n'.join(summaries)
    
    def extract_critical_facts(self, content: str) -> List[str]:
        """
        Extract critical facts from content that must be preserved.
        
        Extracts:
        - Error patterns (error:, failed:, exception:)
        - Decisions (decided, chose, because)
        - File paths modified
        - Task completions
        - Build commands
        """
        facts = []
        
        # Error patterns
        error_patterns = re.findall(
            r'(?:error|failed|exception|bug|fix)[:\s]+([^\n]{20,200})', 
            content, re.IGNORECASE
        )
        for e in error_patterns[:10]:
            facts.append(f"ERROR: {e.strip()}")
        
        # Decision patterns
        decision_patterns = re.findall(
            r'(?:decided|chose|because|reason)[:\s]+([^\n]{20,200})', 
            content, re.IGNORECASE
        )
        for d in decision_patterns[:10]:
            facts.append(f"DECISION: {d.strip()}")
        
        # File modifications
        file_patterns = re.findall(
            r'(?:created|modified|updated|deleted|added)[:\s]+([^\n]*\.[a-z]{2,4}[^\n]{0,100})', 
            content, re.IGNORECASE
        )
        for f in file_patterns[:15]:
            facts.append(f"FILE: {f.strip()}")
        
        # Task completions
        task_patterns = re.findall(
            r'(?:completed|finished|done)[:\s]+(?:task[:\s]+)?([^\n]{10,150})', 
            content, re.IGNORECASE
        )
        for t in task_patterns[:10]:
            facts.append(f"COMPLETED: {t.strip()}")
        
        # Build/run commands
        cmd_patterns = re.findall(
            r'(?:run|build|test|install)[:\s]+`?([^\n`]{10,100})`?', 
            content, re.IGNORECASE
        )
        for c in cmd_patterns[:5]:
            facts.append(f"COMMAND: {c.strip()}")
        
        return facts
    
    def verify_condensation(self, original_content: str, condensed: str, 
                           semantic: 'SemanticSearch', 
                           metrics: Optional['RalphMetrics'] = None) -> Tuple[bool, List[str]]:
        """
        Verify that critical facts from original are preserved in condensed.
        
        Returns:
            (is_valid, list_of_missing_facts)
        """
        # Extract facts from original
        original_facts = self.extract_critical_facts(original_content)
        
        if not original_facts:
            # No critical facts found, accept condensation
            if metrics:
                metrics.record_condense(facts_preserved=0, facts_total=0)
            return True, []
        
        # Check each fact is semantically present in condensed
        missing = []
        preserved = 0
        
        # Split condensed into sentences for comparison
        condensed_lower = condensed.lower()
        
        for fact in original_facts:
            # Check if fact content is present (semantic or literal)
            fact_content = fact.split(':', 1)[-1].strip().lower()
            
            # First try literal match (key words)
            key_words = [w for w in fact_content.split() if len(w) > 4][:5]
            literal_match = sum(1 for w in key_words if w in condensed_lower) >= len(key_words) * 0.5
            
            if literal_match:
                preserved += 1
            else:
                # Try semantic similarity
                sim = semantic.compute_similarity(fact, condensed)
                if sim > 0.3:
                    preserved += 1
                else:
                    missing.append(fact)
        
        total = len(original_facts)
        
        if metrics:
            metrics.record_condense(facts_preserved=preserved, facts_total=total)
        
        # Valid if >70% facts preserved
        is_valid = (preserved / total) >= 0.7 if total > 0 else True
        
        return is_valid, missing
    
    def enhance_condensed_with_missing(self, condensed: str, missing_facts: List[str]) -> str:
        """Add missing critical facts to condensed context."""
        if not missing_facts:
            return condensed
        
        # Add missing facts section
        missing_section = "\n\n## Critical Facts (auto-preserved)\n"
        for fact in missing_facts[:20]:  # Limit to 20
            missing_section += f"- {fact}\n"
        
        return condensed + missing_section


# =============================================================================
# LOGGING
# =============================================================================

def log(message: str):
    """Write to log file."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def log_error(message: str):
    """Log error message."""
    log(f"ERROR: {message}")


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape codes from text.
    
    Handles:
    - Standard CSI sequences (colors, cursor movement)
    - OSC sequences (terminal titles)
    - DCS/PM/APC/SOS sequences
    - Malformed sequences (missing ESC byte)
    """
    if not text:
        return text
    text = text.replace('\r', '')
    # Standard CSI sequences
    text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
    # More CSI patterns
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    # OSC sequences (terminated by BEL or ST)
    text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
    # DCS/PM/APC/SOS
    text = re.sub(r'\x1b[PXY^_][^\x1b\x07]*[\x1b\x07]?', '', text)
    # Malformed sequences (missing ESC - common in redirected output)
    text = re.sub(r'\[[\d;]*[A-Za-z]', '', text)
    return text


def clean_openhands_output(text: str) -> str:
    """
    Clean OpenHands CLI output for display.
    
    Removes:
    - ANSI codes
    - Cursor movement artifacts
    - Progress indicators
    - Duplicate lines
    - Terminal UI noise
    """
    if not text:
        return ""
    
    # Step 1: Remove carriage returns and ANSI
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    text = strip_ansi(text)
    
    # Step 2: Remove cursor movement artifacts
    # Pattern: standalone numbers/semicolons from cursor position codes
    text = re.sub(r'^\d+;\d+H?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+[A-Z]$', '', text, flags=re.MULTILINE)
    
    # Step 3: Remove progress bar artifacts
    text = re.sub(r'[░▒▓█]+', '', text)
    text = re.sub(r'[\|/\-\\]{3,}', '', text)  # Spinner artifacts
    
    # Step 4: Remove common OpenHands UI elements
    ui_noise_patterns = [
        r'^Working \(\d+s.*$',
        r'^.*\| Type your message.*$',
        r'^\s*Ctrl.*$',
        r'^\s*ESC:.*$',
        r'^Tokens:.*$',
        r'^Cost:.*$',
        r'^cache \d+%.*$',
        r'^\d+K \• \$.*$',
        r'^─+$',
        r'^╭.*╮$',
        r'^╰.*╯$',
        r'^│\s*│$',
        r'^\s*\d+\s*$',  # Standalone numbers
        r'^M\s*$',  # Cursor artifact
        r'^\s+$',  # Whitespace only
        r'^Press.*to.*$',
    ]
    
    lines = text.split('\n')
    clean_lines = []
    prev_line = ""
    
    for line in lines:
        line = line.rstrip()
        
        # Skip empty
        if not line:
            if clean_lines and clean_lines[-1] != '':
                clean_lines.append('')
            continue
        
        # Skip noise patterns
        skip = False
        for pattern in ui_noise_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                skip = True
                break
        if skip:
            continue
        
        # Skip exact duplicates (common with progress updates)
        if line == prev_line:
            continue
        
        # Skip lines that are just box drawing characters
        if all(c in '│─┌┐└┘├┤┬┴┼╭╮╰╯║═╔╗╚╝╠╣╦╩╬ ' for c in line):
            continue
        
        clean_lines.append(line)
        prev_line = line
    
    # Remove trailing empty lines
    while clean_lines and clean_lines[-1] == '':
        clean_lines.pop()
    
    # Remove leading empty lines
    while clean_lines and clean_lines[0] == '':
        clean_lines.pop(0)
    
    return '\n'.join(clean_lines)


def parse_openhands_json_events(output: str) -> Tuple[List[Dict], List[str]]:
    """
    Parse OpenHands --headless --json output into events and assistant messages.
    
    Returns:
        Tuple[events, assistant_messages]: Parsed JSON events and extracted messages
    """
    events = []
    assistant_messages = []
    
    def extract_from_event(event: Dict):
        """Extract messages from a parsed event."""
        # Extract assistant messages from MessageEvent
        if event.get('kind') == 'MessageEvent':
            llm_message = event.get('llm_message', {})
            if llm_message.get('role') == 'assistant':
                content_list = llm_message.get('content', [])
                for item in content_list:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text = item.get('text', '')
                        if text:
                            assistant_messages.append(text)
        
        # Extract from ObservationEvent - show what happened
        elif event.get('kind') == 'ObservationEvent':
            obs = event.get('observation', {})
            obs_kind = obs.get('kind', '')
            
            if obs_kind == 'TerminalObservation':
                # Terminal command output
                obs_content = obs.get('content', [])
                for item in obs_content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text = item.get('text', '')
                        if text:
                            # Show truncated output with indication
                            if len(text) > 2000:
                                text = text[:2000] + "\n... (truncated)"
                            assistant_messages.append(f"```\n{text}\n```")
            
            elif obs_kind == 'FileEditorObservation':
                # File edit result - show what was changed
                obs_content = obs.get('content', [])
                for item in obs_content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text = item.get('text', '')
                        if text and len(text) < 3000:
                            assistant_messages.append(f"```\n{text[:2500]}\n```")
        
        # Extract from ActionEvent - the agent's action
        elif event.get('kind') == 'ActionEvent':
            action = event.get('action', {})
            action_kind = action.get('kind', '')
            summary = event.get('summary', '')
            
            # First, show thinking if available
            thought = event.get('thought', [])
            for item in thought:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text = item.get('text', '').strip()
                    if text:
                        assistant_messages.append(text)
            
            # Show reasoning_content if available (some models use this)
            reasoning = event.get('reasoning_content')
            if reasoning:
                assistant_messages.append(f"*Thinking:* {reasoning}")
            
            # Action-specific formatting
            if action_kind == 'TerminalAction':
                cmd = action.get('command', '')
                if cmd and summary:
                    assistant_messages.append(f"**{summary}**\n```bash\n{cmd}\n```")
                elif cmd:
                    assistant_messages.append(f"```bash\n{cmd}\n```")
            
            elif action_kind == 'FileEditorAction':
                path = action.get('path', '')
                command = action.get('command', '')
                
                if command == 'str_replace':
                    old_str = action.get('old_str', '')
                    new_str = action.get('new_str', '')
                    if summary:
                        assistant_messages.append(f"**{summary}**")
                    if old_str and new_str:
                        # Show diff-like format
                        old_preview = old_str[:500] + "..." if len(old_str) > 500 else old_str
                        new_preview = new_str[:500] + "..." if len(new_str) > 500 else new_str
                        assistant_messages.append(f"Editing `{path}`:\n```diff\n- {old_preview}\n+ {new_preview}\n```")
                    elif new_str:
                        new_preview = new_str[:800] + "..." if len(new_str) > 800 else new_str
                        assistant_messages.append(f"Adding to `{path}`:\n```\n{new_preview}\n```")
                
                elif command == 'create':
                    file_text = action.get('file_text', '')
                    if summary:
                        assistant_messages.append(f"**{summary}**")
                    if file_text:
                        preview = file_text[:1000] + "..." if len(file_text) > 1000 else file_text
                        assistant_messages.append(f"Creating `{path}`:\n```\n{preview}\n```")
                
                elif command == 'view':
                    if summary:
                        assistant_messages.append(f"**{summary}**")
                
                else:
                    # Other file operations
                    if summary:
                        assistant_messages.append(f"**{summary}**")
            
            elif action_kind == 'ThinkAction':
                # Model's explicit thinking
                thought_text = action.get('thought', '')
                if thought_text:
                    assistant_messages.append(f"*Thinking:* {thought_text}")
            
            elif action_kind == 'FinishAction':
                # Task completion
                message = action.get('message', '')
                if message:
                    assistant_messages.append(f"**Finished:** {message}")
            
            else:
                # Generic action - just show summary
                if summary:
                    assistant_messages.append(f"**{summary}**")
    
    def clean_json_output(json_text: str) -> str:
        """Clean OpenHands JSON output from spinner pollution and control chars.
        
        OpenHands CLI has issues:
        1. Outputs spinner/progress text INSIDE JSON blocks
        2. Pretty JSON with literal newlines in string values
        
        This function removes spinner lines and fixes control characters.
        """
        # First: remove ANSI escape sequences and spinner lines
        # Pattern: lines containing spinner chars (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) or "Agent is working"
        import re
        
        # Remove lines that are clearly spinner/progress output
        lines = json_text.split('\n')
        clean_lines = []
        for line in lines:
            # Skip lines with spinner characters
            if any(c in line for c in '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'):
                continue
            # Skip lines with "Agent is working" progress
            if 'Agent is working' in line:
                continue
            # Remove ANSI codes from line
            clean_line = re.sub(r'\x1b\[[0-9;]*[mK]', '', line)
            clean_lines.append(clean_line)
        
        json_text = '\n'.join(clean_lines)
        
        # Second: fix control characters in strings
        result = []
        in_string = False
        escape_next = False
        
        for char in json_text:
            if escape_next:
                result.append(char)
                escape_next = False
                continue
            
            if char == '\\':
                result.append(char)
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                result.append(char)
                continue
            
            if in_string:
                # Inside a string - escape control characters
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif ord(char) < 32:  # Other control chars
                    result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    # Check for --JSON Event-- markers (pretty-printed JSON from OpenHands)
    json_marker = "--JSON Event--"
    
    if json_marker in output:
        parts = output.split(json_marker)
        parsed_count = 0
        
        for part in parts[1:]:  # Skip header text before first marker
            # Find first '{' - skip any spinner/progress text before JSON
            json_start = part.find('{')
            if json_start == -1:
                continue
            json_text = part[json_start:]
            
            # Clean JSON output from spinner pollution and control chars
            json_text = clean_json_output(json_text)
            
            # Use JSONDecoder for robust parsing (handles partial/streaming JSON)
            decoder = json.JSONDecoder()
            try:
                event, _ = decoder.raw_decode(json_text)
                events.append(event)
                extract_from_event(event)
                parsed_count += 1
            except (json.JSONDecodeError, ValueError):
                continue
        
        # If JSON parsing failed but we have markers, try regex extraction as fallback
        if parsed_count == 0 and len(parts) > 1:
            # Extract summaries from action events
            summaries = re.findall(r'"summary":\s*"([^"]{3,200})"', output)
            for summary in summaries:
                if summary and summary not in assistant_messages:
                    assistant_messages.append(f"→ {summary}")
    
    if not events:
        # Fallback: line-by-line JSONL parsing
        for line in output.split('\n'):
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                event = json.loads(line)
                events.append(event)
                
                if event.get('kind') == 'MessageEvent':
                    llm_message = event.get('llm_message', {})
                    if llm_message.get('role') == 'assistant':
                        content_list = llm_message.get('content', [])
                        for item in content_list:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text = item.get('text', '')
                                if text:
                                    assistant_messages.append(text)
                
                # Legacy format support
                elif event.get('type') == 'message' and event.get('role') == 'assistant':
                    content = event.get('content', '')
                    if content:
                        assistant_messages.append(content)
                elif event.get('action') == 'message':
                    args = event.get('args', {})
                    content = args.get('content', '')
                    if content:
                        assistant_messages.append(content)
            except json.JSONDecodeError:
                continue
    
    return events, assistant_messages


# =============================================================================
# DOCKER OPERATIONS  
# =============================================================================

class Docker:
    """Docker operations with enhanced error handling."""
    
    @staticmethod
    def is_available() -> bool:
        """Check if docker is available."""
        try:
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def container_exists(name: str) -> bool:
        """Check if container exists."""
        try:
            # Use docker inspect instead of ps filter (more reliable)
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", name],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False
    
    @staticmethod
    def container_running(name: str) -> bool:
        """Check if container is running."""
        try:
            # Use docker inspect instead of ps filter (more reliable)
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", name],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and result.stdout.strip().lower() == "true"
        except Exception:
            return False
    
    @staticmethod
    def start_container(name: str) -> bool:
        """Start stopped container."""
        try:
            result = subprocess.run(
                ["docker", "start", name],
                capture_output=True, timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def stop_container(name: str, timeout: int = 10) -> bool:
        """Stop running container."""
        try:
            result = subprocess.run(
                ["docker", "stop", "-t", str(timeout), name],
                capture_output=True, timeout=timeout + 30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def remove_container(name: str, force: bool = False) -> bool:
        """Remove container."""
        try:
            cmd = ["docker", "rm"]
            if force:
                cmd.append("-f")
            cmd.append(name)
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def get_containers(prefix: str = "openhands-") -> List[Dict]:
        """Get list of containers with given prefix."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={prefix}",
                 "--format", "{{.Names}}\t{{.Status}}\t{{.Size}}"],
                capture_output=True, text=True, timeout=10
            )
            containers = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        containers.append({
                            "name": parts[0],
                            "status": parts[1],
                            "size": parts[2] if len(parts) > 2 else "?"
                        })
            return containers
        except Exception:
            return []
    
    @staticmethod
    def exec_in_container(name: str, command: str, timeout: int = 60) -> Tuple[int, str]:
        """Execute command in container.
        
        SECURITY WARNING: This method executes shell commands. For user-provided
        content, ALWAYS use write_file_safe() which base64-encodes content.
        """
        try:
            result = subprocess.run(
                ["docker", "exec", name, "bash", "-c", command],
                capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "Timeout"
        except Exception as e:
            return -1, str(e)
    
    @staticmethod
    def write_file_safe(container: str, path: str, content: str) -> bool:
        """Write content safely using base64 to prevent shell injection.
        
        SECURITY FIX: Always base64-encode user content to prevent shell injection
        via special characters like backticks, $(), semicolons, etc.
        Use shlex.quote() for paths to prevent path traversal attacks.
        
        Args:
            container: Container name
            path: Target file path (will be shell-quoted)
            content: Content to write (will be base64 encoded)
        
        Returns:
            True if successful, False otherwise
        """
        encoded = base64.b64encode(content.encode()).decode()
        # SECURITY: Single quotes prevent shell expansion, shlex.quote prevents path injection
        quoted_path = shlex.quote(path)
        code, _ = Docker.exec_in_container(
            container,
            f"printf '%s' '{encoded}' | base64 -d > {quoted_path}",
            timeout=10
        )
        return code == 0
    
    @staticmethod
    def pull_image(image: str = None) -> bool:
        """Pull docker image."""
        if image is None:
            image = RUNTIME_IMAGE
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                timeout=600  # 10 minutes
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def create_persistent_container(project: Project) -> bool:
        """Create persistent container if not exists."""
        name = project.container_name
        
        if Docker.container_exists(name):
            return True
        
        # Check if image exists locally
        try:
            result = subprocess.run(
                ["docker", "images", "-q", RUNTIME_IMAGE],
                capture_output=True, text=True, timeout=10
            )
            if not result.stdout.strip():
                print(f"  Image {RUNTIME_IMAGE} not found locally")
                print("  Pulling image (this may take a few minutes)...")
                pull_result = subprocess.run(
                    ["docker", "pull", RUNTIME_IMAGE],
                    timeout=600
                )
                if pull_result.returncode != 0:
                    print("[ERROR] Failed to pull image")
                    return False
                print("  Image pulled")
        except Exception as e:
            print(f"[ERROR] Image check failed: {e}")
            return False
        
        # Ensure directories exist
        project.config_dir.mkdir(parents=True, exist_ok=True)
        project.workspace.mkdir(parents=True, exist_ok=True)
        project.openhands_dir.mkdir(parents=True, exist_ok=True)
        
        # Build docker create command
        # config_dir is mounted to /root, so agent_settings.json should be at /root/.openhands/agent_settings.json
        cmd = [
            "docker", "create",
            "--name", name,
            "-v", "/var/run/docker.sock:/var/run/docker.sock",
            "-v", f"{project.config_dir}:/root",
            "-v", f"{project.workspace}:/workspace",
            "-w", "/workspace",
            "-e", f"PATH={CONTAINER_PATH}",
            "-e", "CARGO_HOME=/root/.cargo",
            "-e", "RUSTUP_HOME=/root/.rustup",
            "-e", "UV_CACHE_DIR=/root/.cache/uv",
            "-e", "UV_TOOL_DIR=/root/.local/share/uv/tools",
            "-e", "UV_TOOL_BIN_DIR=/root/.local/bin",
            "-e", "PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring",
            "-e", "RUNTIME=local",
            "-e", "LOG_ALL_EVENTS=true",
            "--add-host", "host.docker.internal:host-gateway",
            "-p", "3000",
            RUNTIME_IMAGE,
            "sleep", "infinity"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                log_error(f"Failed to create container: {result.stderr}")
                print(f"[ERROR] Docker create failed: {result.stderr}")
                return False
            return True
        except Exception as e:
            log_error(f"Exception creating container: {e}")
            print(f"[ERROR] Exception: {e}")
            return False
    
    @staticmethod
    def setup_container(project: Project) -> bool:
        """One-time setup: install OpenHands and MCP packages."""
        name = project.container_name
        setup_marker = "/root/.openhands/.setup_done"
        mcp_marker = "/root/.openhands/.mcp_warmup_done"
        mcp_servers_file = "/root/.openhands/mcp_servers.json"
        
        # Step 1: Basic setup
        code, output = Docker.exec_in_container(name, f"test -f {setup_marker} && echo 'done'", timeout=5)
        if "done" not in output:
            print("* First-time setup (installing OpenHands)...")
            
            setup_script = f'''
{OPENHANDS_INSTALL_SCRIPT}

mkdir -p /root/.openhands
touch {setup_marker}
echo "[OK] OpenHands setup complete!"
'''
            code, output = Docker.exec_in_container(name, setup_script, timeout=600)
            print(output)
            if code != 0:
                return False
        
        # Step 1.5: Install nano
        code, _ = Docker.exec_in_container(name, "command -v nano", timeout=5)
        if code != 0:
            print("* Installing nano...")
            Docker.exec_in_container(name, "apt-get update -qq && apt-get install -y -qq nano", timeout=120)
        
        # Step 1.6: Copy mcp_servers.json to container if exists locally
        local_mcp = project.config_dir / "mcp_servers.json"
        if local_mcp.exists():
            print("* Copying MCP config to container...")
            cp_result = subprocess.run(
                ["docker", "cp", str(local_mcp), f"{name}:/root/.openhands/mcp_servers.json"],
                capture_output=True, text=True, timeout=10
            )
            if cp_result.returncode == 0:
                print("  MCP config copied")
            else:
                print(f"  [WARNING] Failed to copy MCP config: {cp_result.stderr}")
        
        # Step 2: MCP warmup (with skip option)
        code, output = Docker.exec_in_container(name, f"test -f {mcp_marker} && echo 'done'", timeout=5)
        mcp_warmup_done = "done" in output
        
        code2, output2 = Docker.exec_in_container(name, f"test -f {mcp_servers_file} && echo 'exists'", timeout=5)
        mcp_exists = "exists" in output2
        
        if mcp_exists and not mcp_warmup_done:
            print("* Installing MCP servers and dependencies...")
            print("  This may take 2-5 minutes. Press Ctrl+C to skip MCP setup.")
            print("  (You can install MCP later via project settings)")
            
            warmup_script = f'''
{MCP_WARMUP_SCRIPT}

touch {mcp_marker}
'''
            # Run with progress updates
            stop_progress = threading.Event()
            
            def show_progress():
                elapsed = 0
                while not stop_progress.is_set() and elapsed < 300:
                    time.sleep(10)
                    elapsed += 10
                    if not stop_progress.is_set():
                        print(f"  ... still installing ({elapsed}s)")
            
            progress_thread = threading.Thread(target=show_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            try:
                code, output = Docker.exec_in_container(name, warmup_script, timeout=300)
                stop_progress.set()
                progress_thread.join(timeout=1)
                
                if code != 0:
                    print("  [!] MCP warmup had issues, but continuing without MCP...")
                else:
                    print("  MCP installed successfully!")
                    if output:
                        print(output[-1000:] if len(output) > 1000 else output)
            except KeyboardInterrupt:
                stop_progress.set()
                print("\n  [!] MCP setup interrupted by user")
                print("  Continuing without MCP. You can install MCP later via project settings.")
                # Mark as done so we don't try again
                Docker.exec_in_container(name, f"touch {mcp_marker}", timeout=5)
        
        # Step 3: Setup auto-start MCP on container restart
        autostart_script = '''
# Add MCP auto-start to profile if not already there
if ! grep -q "mcp_autostart.sh" /root/.bashrc 2>/dev/null; then
    echo "" >> /root/.bashrc
    echo "# Auto-start MCP gateways on container start" >> /root/.bashrc
    echo "if [ -f /root/.openhands/mcp_autostart.sh ]; then" >> /root/.bashrc
    echo "    bash /root/.openhands/mcp_autostart.sh > /tmp/mcp_autostart.log 2>&1 &" >> /root/.bashrc
    echo "fi" >> /root/.bashrc
fi

# Create the autostart script
cat > /root/.openhands/mcp_autostart.sh << 'AUTOEOF'
#!/bin/bash
# MCP Auto-start script - runs when container starts

export PATH="/root/.local/bin:/root/.cargo/bin:/root/.bun/bin:$PATH"
MCP_SERVERS="/root/.openhands/mcp_servers.json"
GATEWAY_PIDS="/root/.openhands/.gateway_pids"

# Wait a bit for system to stabilize
sleep 2

# Check if already running
if [ -f "$GATEWAY_PIDS" ]; then
    running=0
    while read pid; do
        if kill -0 $pid 2>/dev/null; then
            running=$((running + 1))
        fi
    done < "$GATEWAY_PIDS"
    
    if [ $running -gt 0 ]; then
        echo "MCP gateways already running ($running)"
        exit 0
    fi
fi

# Start MCP gateways if config exists
if [ -f "$MCP_SERVERS" ] && [ -f /root/.openhands/.gateway_state ]; then
    echo "Auto-starting MCP gateways..."
    # The actual gateway script will be called by ensure_container_running
    # This is just a fallback marker
    touch /root/.openhands/.mcp_autostart_requested
fi

# QMD setup if enabled but not installed
if [ -f /workspace/.qmd_enabled ] && [ ! -f /workspace/.qmd_installed ]; then
    echo "QMD enabled but not installed, running setup..."
    if [ -f /workspace/.qmd_setup.sh ]; then
        bash /workspace/.qmd_setup.sh >> /tmp/qmd_setup.log 2>&1 && touch /workspace/.qmd_installed
        echo "QMD setup complete"
    fi
fi

# QMD reindex if installed (background, low priority)
if [ -f /workspace/.qmd_installed ] && [ -f /workspace/.qmd_reindex.sh ]; then
    nice -n 19 bash /workspace/.qmd_reindex.sh >> /tmp/qmd_reindex.log 2>&1 &
fi
AUTOEOF

chmod +x /root/.openhands/mcp_autostart.sh
'''
        Docker.exec_in_container(name, autostart_script, timeout=30)
        
        return True
    
    @staticmethod
    def start_mcp_gateways(project: Project) -> bool:
        """Start or restart MCP gateways."""
        name = project.container_name
        
        # Check if mcp_servers.json exists first
        code, _ = Docker.exec_in_container(name, "test -f /root/.openhands/mcp_servers.json", timeout=5)
        if code != 0:
            return True  # No MCP config, nothing to do
        
        # Use longer timeout (3 min + buffer) for slow MCP like memory
        # Show progress while waiting
        stop_progress = threading.Event()
        
        def show_gateway_progress():
            elapsed = 0
            while not stop_progress.is_set() and elapsed < 240:
                time.sleep(10)
                elapsed += 10
                if not stop_progress.is_set():
                    print(f"  ... starting gateways ({elapsed}s)")
        
        progress_thread = threading.Thread(target=show_gateway_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            print("  Running gateway script (timeout: 300s)...")
            code, output = Docker.exec_in_container(name, MCP_GATEWAY_SCRIPT, timeout=300)
            stop_progress.set()
            progress_thread.join(timeout=1)
            
            if code != 0:
                print(f"  [!] MCP gateway start failed with exit code: {code}")
                if output:
                    print(f"  Output:\n{output[-2000:]}")
                else:
                    print("  No output captured")
                # Try to get gateway logs
                try:
                    log_code, log_output = Docker.exec_in_container(name, "ls -la /tmp/gateway_*.log 2>/dev/null; cat /tmp/gateway_*.log 2>/dev/null | tail -100", timeout=10)
                    if log_output:
                        print(f"  Gateway logs:\n{log_output}")
                except Exception as e:
                    print(f"  Could not get gateway logs: {e}")
            else:
                print("  Gateways started successfully!")
                if output:
                    print(output[-1500:] if len(output) > 1500 else output)
        except KeyboardInterrupt:
            stop_progress.set()
            print("\n  [!] Gateway start interrupted")
            print("  Continuing without MCP.")
        
        return True
    
    @staticmethod
    def restart_mcp_gateways(project: Project) -> bool:
        """Force restart all MCP gateways (kill and restart)."""
        name = project.container_name
        
        print("* Force restarting MCP gateways...")
        
        # Kill all existing gateways
        kill_script = '''
pkill -f "supergateway" 2>/dev/null || true
pkill -f "mcp-server" 2>/dev/null || true
pkill -f "mcp-memory" 2>/dev/null || true
sleep 2
rm -f /root/.openhands/.gateway_pids
rm -f /root/.openhands/.gateway_state
echo "All gateways stopped"
'''
        Docker.exec_in_container(name, kill_script, timeout=30)
        
        # Now start fresh
        return Docker.start_mcp_gateways(project)
    
    @staticmethod
    def upgrade_mcp_servers(project: Project, new_config: Optional[str] = None) -> bool:
        """Upgrade MCP servers: install new packages, remove old ones, restart gateways.
        
        Args:
            project: Project instance
            new_config: Optional new mcp_servers.json content to write
        
        Returns:
            True if successful
        """
        name = project.container_name
        
        print("* Upgrading MCP servers...")
        
        # If new config provided, write it
        if new_config:
            escaped_config = new_config.replace("'", "'\\''")
            write_script = f"echo '{escaped_config}' > /root/.openhands/mcp_servers.json"
            Docker.exec_in_container(name, write_script, timeout=10)
            print("  New MCP config written")
        
        # Get list of required packages from new config
        upgrade_script = '''
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
MCP_SERVERS="/root/.openhands/mcp_servers.json"

if [ ! -f "$MCP_SERVERS" ]; then
    echo "ERROR: mcp_servers.json not found"
    exit 1
fi

echo "Analyzing MCP requirements..."

# Get current installed packages (uvx)
current_uvx=$(ls -1 /root/.local/share/uv/tools/ 2>/dev/null || echo "")

# Get required packages from config
required_uvx=$(jq -r '.mcpServers // {} | to_entries[] | select(.value.command == "uvx") | .value.args | map(select(startswith("-") | not)) | .[0] // empty' "$MCP_SERVERS" 2>/dev/null | sort -u)
required_npx=$(jq -r '.mcpServers // {} | to_entries[] | select(.value.command == "npx") | .value.args | map(select(startswith("-") | not)) | .[0] // empty' "$MCP_SERVERS" 2>/dev/null | sort -u)

echo "Required uvx packages: $required_uvx"
echo "Required npx packages: $required_npx"

# Install missing uvx packages
echo ""
echo "Installing/updating uvx packages..."
for pkg in $required_uvx; do
    if [ -n "$pkg" ]; then
        echo "  -> $pkg"
        timeout 120 uv tool install "$pkg" 2>&1 | tail -3 || echo "    (already installed or timeout)"
    fi
done

# Install missing npx packages
echo ""
echo "Installing/updating npx packages..."
for pkg in $required_npx; do
    if [ -n "$pkg" ]; then
        echo "  -> $pkg"
        timeout 180 npm install -g "$pkg" 2>&1 | tail -3 || echo "    (already installed or timeout)"
    fi
done

echo ""
echo "MCP packages updated!"
'''
        code, output = Docker.exec_in_container(name, upgrade_script, timeout=300)
        print(output[-2000:] if len(output) > 2000 else output)
        
        # Force restart gateways with new config
        print("* Restarting gateways with new configuration...")
        return Docker.restart_mcp_gateways(project)
    
    @staticmethod
    def check_mcp_gateways(project: Project) -> Tuple[int, int, int]:
        """Check how many MCP gateways are running and healthy.
        
        Returns: (running_count, total_count, healthy_count)
        """
        name = project.container_name
        # Check by actual running processes, not stale PID file
        check_script = '''
MCP_SERVERS="/root/.openhands/mcp_servers.json"
running=0
total=0
healthy=0

# Count expected servers from config
if [ -f "$MCP_SERVERS" ]; then
    total=$(jq -r '.mcpServers | keys | length' "$MCP_SERVERS" 2>/dev/null || echo 0)
fi

# Count running supergateway processes
running=$(pgrep -c -f "supergateway" 2>/dev/null || echo 0)

# Health check each port - use HTTP response code only (SSE streams forever)
port=8001
while [ $port -le $((8000 + total)) ]; do
    # Use curl -o /dev/null to discard body, -w to get status code
    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 1 "http://localhost:$port/sse" 2>/dev/null)
    if [ "$status" = "200" ]; then
        healthy=$((healthy + 1))
    fi
    port=$((port + 1))
done

echo "$running $total $healthy"
'''
        code, output = Docker.exec_in_container(name, check_script, timeout=15)
        if code == 0:
            parts = output.strip().split()
            if len(parts) >= 3:
                try:
                    return int(parts[0]), int(parts[1]), int(parts[2])
                except Exception:
                    pass
            elif len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1]), 0
                except Exception:
                    pass
        return 0, 0, 0
    
    @staticmethod
    def ensure_container_running(project: Project, restart_gateways: bool = True) -> bool:
        """Ensure container exists, is running, and is setup."""
        name = project.container_name
        
        # Check Docker is working
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            if result.returncode != 0:
                print("[ERROR] Docker is not running!")
                return False
        except Exception as e:
            print(f"[ERROR] Docker check failed: {e}")
            return False
        
        # Ensure image exists (in case it was deleted after project creation)
        try:
            result = subprocess.run(
                ["docker", "images", "-q", RUNTIME_IMAGE],
                capture_output=True, text=True, timeout=10
            )
            if not result.stdout.strip():
                print(f"* Image {RUNTIME_IMAGE} not found, pulling...")
                pull_result = subprocess.run(
                    ["docker", "pull", RUNTIME_IMAGE],
                    timeout=600
                )
                if pull_result.returncode != 0:
                    print("[ERROR] Failed to pull image")
                    return False
                print("  Image pulled")
        except Exception as e:
            print(f"[ERROR] Image check failed: {e}")
            return False
        
        # Create if not exists
        if not Docker.container_exists(name):
            print("* Creating container...")
            if not Docker.create_persistent_container(project):
                print("[ERROR] Failed to create container")
                return False
            print("  Container created")
        
        # Start if not running
        if not Docker.container_running(name):
            print("* Starting container...")
            if not Docker.start_container(name):
                print("[ERROR] Failed to start container")
                return False
            print("  Container started")
        
        # Wait for container to be ready
        print("* Waiting for container to be ready...")
        last_error = ""
        for i in range(30):
            code, output = Docker.exec_in_container(name, "true", timeout=5)
            if code == 0:
                break
            last_error = output.strip() if output else f"exit code {code}"
            if i % 5 == 4:  # Every 5 attempts
                print(f"  Still waiting... (attempt {i+1}/30, last error: {last_error[:50]})")
            time.sleep(1)
        else:
            print(f"[ERROR] Container not responding after 30s: {last_error}")
            return False
        
        # Check who we are and where $HOME is
        code, whoami = Docker.exec_in_container(name, "whoami", timeout=5)
        code, home = Docker.exec_in_container(name, "echo $HOME", timeout=5)
        print(f"  Container user: {whoami.strip()}, HOME: {home.strip()}")
        
        # Verify agent_settings.json exists in container with proper permissions
        # Check both /root/.openhands and $HOME/.openhands
        for check_path in ["/root/.openhands", home.strip() + "/.openhands"]:
            # Check file exists and permissions
            code, output = Docker.exec_in_container(
                name, 
                f"ls -la {check_path}/agent_settings.json 2>&1",
                timeout=5
            )
            if code == 0:
                print(f"  File info: {output.strip()}")
                
                # Check if readable and is text file
                code, output = Docker.exec_in_container(
                    name,
                    f"file {check_path}/agent_settings.json",
                    timeout=5
                )
                print(f"  File type: {output.strip()}")
                
                code, output = Docker.exec_in_container(
                    name,
                    f"cat {check_path}/agent_settings.json | head -1",
                    timeout=5
                )
                if code != 0:
                    print(f"[!] File exists but NOT READABLE! Fixing permissions...")
                    Docker.exec_in_container(name, f"chmod 644 {check_path}/agent_settings.json", timeout=5)
                    Docker.exec_in_container(name, f"chown root:root {check_path}/agent_settings.json", timeout=5)
                
                # HIGH PRIORITY FIX: Validate required fields (llm.model, llm.api_key)
                # Use a simpler validation approach to avoid f-string issues
                validate_cmd = (
                    "cat " + check_path + "/agent_settings.json | python3 -c '\n"
                    "import sys, json\n"
                    "try:\n"
                    "    d = json.load(sys.stdin)\n"
                    "    llm = d.get(\"llm\", {})\n"
                    "    model = llm.get(\"model\", \"\")\n"
                    "    api_key = llm.get(\"api_key\", \"\")\n"
                    "    errors = []\n"
                    "    if not model:\n"
                    "        errors.append(\"missing llm.model\")\n"
                    "    if not api_key:\n"
                    "        errors.append(\"missing llm.api_key\")\n"
                    "    if errors:\n"
                    "        print(\"JSON_INVALID\", \",\".join(errors))\n"
                    "    else:\n"
                    "        print(\"JSON_OK\", model)\n"
                    "except Exception as e:\n"
                    "    print(\"JSON_ERROR\", str(e))\n"
                    "' 2>&1"
                )
                code, output = Docker.exec_in_container(name, validate_cmd, timeout=5)
                
                config_path = check_path
                print(f"  Found agent_settings.json at: {config_path}/")
                
                if "JSON_OK" in output:
                    parts = output.strip().split()
                    model = parts[1] if len(parts) > 1 else "unknown"
                    print(f"  agent_settings.json: OK (model: {model})")
                elif "JSON_INVALID" in output:
                    parts = output.strip().split(maxsplit=1)
                    errors = parts[1] if len(parts) > 1 else "unknown error"
                    print(f"[!] WARNING: agent_settings.json missing required fields: {errors}")
                    print("    Required: llm.model, llm.api_key")
                else:
                    print(f"[!] WARNING: agent_settings.json check failed: {output.strip()}")
                break
        else:
            print("[!] WARNING: agent_settings.json not found in container!")
            print("    Searched in: /root/.openhands and $HOME/.openhands")
            print("    OpenHands will show setup wizard.")
            print("    Put your agent_settings.json to: config/.openhands/")
        
        # Run one-time setup if needed
        if not Docker.setup_container(project):
            print("[!] Setup had issues, but continuing...")
        
        # Verify mcp_servers.json was copied
        code, _ = Docker.exec_in_container(
            name, 
            "test -f /root/.openhands/mcp_servers.json", 
            timeout=5
        )
        if code == 0:
            print("  mcp_servers.json: found in container")
        else:
            print("  [!] mcp_servers.json: NOT found in container")
        
        # Ensure MCP is fully setup and running
        if restart_gateways:
            try:
                # Check if mcp_servers.json exists
                code, _ = Docker.exec_in_container(
                    name, 
                    "test -f /root/.openhands/mcp_servers.json", 
                    timeout=5
                )
                
                if code == 0:
                    # Check if warmup was done
                    warmup_code, _ = Docker.exec_in_container(
                        name,
                        "test -f /root/.openhands/.mcp_warmup_done",
                        timeout=5
                    )
                    
                    if warmup_code != 0:
                        # Warmup not done - run it now
                        print("* MCP warmup not done, running now...")
                        mcp_marker = "/root/.openhands/.mcp_warmup_done"
                        warmup_script = f'''
{MCP_WARMUP_SCRIPT}

touch {mcp_marker}
'''
                        code, output = Docker.exec_in_container(name, warmup_script, timeout=300)
                        if code == 0:
                            print("  MCP warmup completed ✓")
                        else:
                            print(f"  [!] MCP warmup failed, continuing...")
                    
                    # Check current gateway status
                    running, total, healthy = Docker.check_mcp_gateways(project)
                    
                    if total > 0 and healthy == total:
                        # All gateways healthy
                        print(f"  MCP gateways: {healthy}/{total} healthy ✓")
                    else:
                        # Need to start/restart gateways
                        if total > 0:
                            print(f"* MCP gateways need restart ({healthy}/{total} healthy)...")
                        else:
                            print("* Starting MCP gateways...")
                        
                        Docker.start_mcp_gateways(project)
                        
                        # Verify after start
                        time.sleep(2)
                        running, total, healthy = Docker.check_mcp_gateways(project)
                        if healthy == total and total > 0:
                            print(f"  MCP gateways started: {healthy}/{total} ✓")
                        elif total > 0:
                            print(f"  [!] MCP gateways: only {healthy}/{total} healthy")
            except Exception as e:
                print(f"  [!] MCP check failed: {e}, continuing...")
        
        return True
    
    @staticmethod
    def setup_ralph_daemon(project: Project) -> bool:
        """Setup Ralph daemon and watchdog inside container."""
        name = project.container_name
        
        # Copy ralph_daemon.py to container
        daemon_src = RALPH_PROMPTS_DIR / "ralph_daemon.py"
        if not daemon_src.exists():
            print(f"[!] ralph_daemon.py not found at {daemon_src}")
            return False
        
        # Create .ralph directories in container
        Docker.exec_in_container(name, "mkdir -p /workspace/.ralph/prompts /workspace/.ralph/iterations", timeout=5)
        
        # Copy daemon script
        try:
            subprocess.run(
                ["docker", "cp", str(daemon_src), f"{name}:/workspace/.ralph/ralph_daemon.py"],
                check=True, timeout=10
            )
            Docker.exec_in_container(name, "chmod +x /workspace/.ralph/ralph_daemon.py", timeout=5)
        except Exception as e:
            print(f"[!] Failed to install ralph daemon: {e}")
            return False
        
        # Pre-install dependencies (sentence-transformers is ~500MB, takes time)
        print("  Checking daemon dependencies...")
        code, output = Docker.exec_in_container(
            name,
            "python3 -c 'import sentence_transformers' 2>/dev/null && echo 'INSTALLED' || echo 'MISSING'",
            timeout=10
        )
        if "MISSING" in output:
            # Check disk space first
            _, df_out = Docker.exec_in_container(name, "df -h / | tail -1", timeout=5)
            print(f"  Disk space: {df_out.strip()}")
            
            print("  Installing sentence-transformers (this may take 2-5 minutes)...")
            code, output = Docker.exec_in_container(
                name,
                "pip install sentence-transformers numpy --break-system-packages 2>&1",
                timeout=600  # 10 minutes for large packages
            )
            if code != 0:
                if "No space left" in output:
                    print(f"  [!] NOT ENOUGH DISK SPACE for sentence-transformers (~1GB needed)")
                    print(f"  [!] Free up space with: docker system prune -a")
                else:
                    print(f"  [!] Failed to install dependencies: {output[:500]}")
                return False
            print("  Dependencies installed")
        else:
            print("  Dependencies already installed")
        
        # Copy watchdog script
        watchdog_src = RALPH_PROMPTS_DIR / "ralph_watchdog.sh"
        if watchdog_src.exists():
            try:
                subprocess.run(
                    ["docker", "cp", str(watchdog_src), f"{name}:/workspace/.ralph/ralph_watchdog.sh"],
                    check=True, timeout=10
                )
                Docker.exec_in_container(name, "chmod +x /workspace/.ralph/ralph_watchdog.sh", timeout=5)
            except Exception:
                pass  # Non-critical
        
        # Copy prompt templates if they exist
        for prompt_file in RALPH_PROMPTS_DIR.glob("*.md"):
            try:
                subprocess.run(
                    ["docker", "cp", str(prompt_file), f"{name}:/workspace/.ralph/prompts/"],
                    check=True, timeout=10
                )
            except Exception:
                pass  # Non-critical
        
        print("  Ralph daemon installed in container")
        return True
    
    @staticmethod
    def sync_ralph_prompts(project: Project) -> bool:
        """Sync global prompt templates from host to container."""
        name = project.container_name
        
        # Create global prompts directory
        Docker.exec_in_container(name, "mkdir -p /workspace/.ralph/prompts/global", timeout=5)
        
        # Copy all .md files from templates/ralph/ to container
        for prompt_file in RALPH_PROMPTS_DIR.glob("*.md"):
            try:
                subprocess.run(
                    ["docker", "cp", str(prompt_file), f"{name}:/workspace/.ralph/prompts/global/"],
                    check=True, timeout=10, capture_output=True
                )
            except Exception:
                pass  # Non-critical
        
        return True
    
    @staticmethod
    def start_ralph_daemon(project: Project) -> bool:
        """Start Ralph daemon inside container with watchdog."""
        name = project.container_name
        
        # Sync prompts from host
        Docker.sync_ralph_prompts(project)
        
        # Check if already running
        code, output = Docker.exec_in_container(
            name, "pgrep -f 'ralph_daemon.py'", timeout=5
        )
        if code == 0 and output.strip():
            print(f"  Ralph daemon already running (PID: {output.strip().split()[0]})")
            return True
        
        # Ensure daemon is installed
        code, _ = Docker.exec_in_container(
            name, "test -f /workspace/.ralph/ralph_daemon.py", timeout=5
        )
        if code != 0:
            if not Docker.setup_ralph_daemon(project):
                return False
        
        # Clear old log
        Docker.exec_in_container(
            name, "> /workspace/.ralph/ralph_daemon.log", timeout=5
        )
        
        # Start daemon with setsid for session independence (survives docker exec exit)
        start_cmd = """
cd /workspace
setsid python3 -u /workspace/.ralph/ralph_daemon.py >> /workspace/.ralph/ralph_daemon.log 2>&1 &
echo $! > /workspace/.ralph/ralph_daemon.pid
sleep 2
pgrep -f 'ralph_daemon.py' && echo 'STARTED' || echo 'FAILED'
"""
        code, output = Docker.exec_in_container(name, start_cmd, timeout=15)
        
        if "FAILED" in output:
            _, log = Docker.exec_in_container(
                name, "cat /workspace/.ralph/ralph_daemon.log", timeout=5
            )
            print(f"  [!] Daemon failed to start. Log:\n{log[:500]}")
            return False
        
        # Setup watchdog cron
        Docker.exec_in_container(
            name,
            '''(crontab -l 2>/dev/null | grep -v ralph_watchdog; echo "* * * * * /workspace/.ralph/ralph_watchdog.sh >> /workspace/.ralph/watchdog.log 2>&1") | crontab -''',
            timeout=10
        )
        Docker.exec_in_container(name, "service cron start 2>/dev/null || true", timeout=5)
        
        # Verify daemon is running
        code, output = Docker.exec_in_container(
            name, "pgrep -f 'ralph_daemon.py'", timeout=5
        )
        if code == 0 and output.strip():
            print(f"  Ralph daemon started (PID: {output.strip().split()[0]})")
            return True
        
        # Show log on failure
        _, log = Docker.exec_in_container(
            name, "tail -20 /workspace/.ralph/ralph_daemon.log", timeout=5
        )
        print(f"  [!] Daemon not running. Log:\n{log}")
        return False
    
    @staticmethod
    def stop_ralph_daemon(project: Project) -> bool:
        """Stop Ralph daemon and watchdog inside container."""
        name = project.container_name
        
        # Remove watchdog cron first
        Docker.exec_in_container(
            name,
            '''crontab -l 2>/dev/null | grep -v ralph_watchdog | crontab - 2>/dev/null || true''',
            timeout=10
        )
        
        # Kill daemon process
        code, _ = Docker.exec_in_container(
            name,
            "pkill -f 'ralph_daemon.py' || true",
            timeout=10
        )
        
        # Remove PID file
        Docker.exec_in_container(
            name,
            "rm -f /workspace/.ralph/ralph_daemon.pid",
            timeout=5
        )
        
        print("  Ralph daemon and watchdog stopped")
        return True
    
    @staticmethod
    def is_ralph_daemon_running(project: Project) -> bool:
        """Check if Ralph daemon is running in container."""
        name = project.container_name
        
        code, output = Docker.exec_in_container(
            name,
            "pgrep -f 'ralph_daemon.py'",
            timeout=5
        )
        
        return code == 0 and bool(output.strip())
    
    @staticmethod
    def get_ralph_status(project: Project) -> dict:
        """Get Ralph status from inside container."""
        name = project.container_name
        
        result = {
            "daemon_running": False,
            "status": "unknown",
            "iteration": 0,
            "heartbeat_age": -1
        }
        
        # Check daemon
        result["daemon_running"] = Docker.is_ralph_daemon_running(project)
        
        # Read config
        code, output = Docker.exec_in_container(
            name,
            "cat /workspace/.ralph/config.json 2>/dev/null || echo '{}'",
            timeout=5
        )
        if code == 0:
            try:
                config = json.loads(output)
                result["status"] = config.get("status", "unknown")
                result["iteration"] = config.get("currentIteration", 0)
            except Exception:
                pass
        
        # Check heartbeat
        code, output = Docker.exec_in_container(
            name,
            "cat /workspace/.ralph/heartbeat 2>/dev/null || echo '0'",
            timeout=5
        )
        if code == 0:
            try:
                hb_time = int(output.strip())
                if hb_time > 0:
                    result["heartbeat_age"] = int(time.time()) - hb_time
            except Exception:
                pass
        
        return result
    
    # =========================================================================
    # FILE OPERATIONS VIA DOCKER EXEC
    # All .ralph file operations go through container to avoid permission issues
    # =========================================================================
    
    @staticmethod
    def read_file(container: str, path: str) -> Optional[str]:
        """Read file content from container. Returns None if file doesn't exist."""
        code, output = Docker.exec_in_container(
            container,
            f"cat '{path}' 2>/dev/null",
            timeout=5
        )
        return output if code == 0 else None
    
    @staticmethod
    def write_file(container: str, path: str, content: str) -> bool:
        """Write content to file in container."""
        # Use base64 to safely transfer content with special chars
        encoded = base64.b64encode(content.encode()).decode()
        code, _ = Docker.exec_in_container(
            container,
            f"echo '{encoded}' | base64 -d > '{path}'",
            timeout=10
        )
        return code == 0
    
    @staticmethod
    def file_exists(container: str, path: str) -> bool:
        """Check if file exists in container."""
        code, _ = Docker.exec_in_container(
            container,
            f"test -f '{path}'",
            timeout=5
        )
        return code == 0
    
    @staticmethod
    def dir_exists(container: str, path: str) -> bool:
        """Check if directory exists in container."""
        code, _ = Docker.exec_in_container(
            container,
            f"test -d '{path}'",
            timeout=5
        )
        return code == 0
    
    @staticmethod
    def mkdir(container: str, path: str) -> bool:
        """Create directory in container."""
        code, _ = Docker.exec_in_container(
            container,
            f"mkdir -p '{path}'",
            timeout=5
        )
        return code == 0
    
    @staticmethod
    def read_json(container: str, path: str, default: Any = None) -> Any:
        """Read JSON file from container."""
        content = Docker.read_file(container, path)
        if content is None:
            return default
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return default
    
    @staticmethod
    def write_json(container: str, path: str, data: Any, indent: int = 2) -> bool:
        """Write JSON data to file in container atomically.
        
        Uses temp file + rename for atomic write to prevent corruption
        if multiple processes write simultaneously.
        
        SECURITY FIX: Use UUID instead of PID for temp file names to prevent
        race conditions when different processes (daemon inside container, TUI 
        outside) have the same PID in their respective namespaces.
        """
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        encoded = base64.b64encode(content.encode()).decode()
        
        # SECURITY FIX: Use UUID for unique temp name instead of PID
        # This prevents race conditions across different PID namespaces
        tmp_path = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
        code, _ = Docker.exec_in_container(
            container,
            f"echo '{encoded}' | base64 -d > '{tmp_path}' && mv '{tmp_path}' '{path}'",
            timeout=10
        )
        return code == 0
    
    @staticmethod
    def append_file(container: str, path: str, content: str) -> bool:
        """Append content to file in container."""
        encoded = base64.b64encode(content.encode()).decode()
        code, _ = Docker.exec_in_container(
            container,
            f"echo '{encoded}' | base64 -d >> '{path}'",
            timeout=10
        )
        return code == 0
    
    @staticmethod
    def delete_file(container: str, path: str) -> bool:
        """Delete file in container."""
        code, _ = Docker.exec_in_container(
            container,
            f"rm -f '{path}'",
            timeout=5
        )
        return code == 0
    
    @staticmethod
    def list_files(container: str, path: str, pattern: str = "*") -> List[str]:
        """List files in container directory."""
        code, output = Docker.exec_in_container(
            container,
            f"ls -1 '{path}'/{pattern} 2>/dev/null || true",
            timeout=5
        )
        if code != 0 or not output.strip():
            return []
        return [f for f in output.strip().split('\n') if f]
    
    @staticmethod
    def run_openhands_session(
        project: Project,
        resume_session: Optional[str] = None,
        task: Optional[str] = None
    ) -> List[str]:
        """Build command to run OpenHands session via docker exec."""
        name = project.container_name
        
        if resume_session:
            oh_cmd = f'openhands --always-approve --resume "{resume_session}"'
        elif task:
            escaped_task = task.replace('"', '\\"').replace("'", "'\\''")
            oh_cmd = f"openhands --always-approve -t '{escaped_task}'"
        else:
            oh_cmd = 'openhands --always-approve'
        
        # Ensure config is readable before starting
        pre_cmd = '''
# Ensure .openhands directory and config exist and are readable
mkdir -p ~/.openhands
if [ -f /root/.openhands/agent_settings.json ]; then
    cp /root/.openhands/agent_settings.json ~/.openhands/agent_settings.json 2>/dev/null || true
    chmod 644 ~/.openhands/agent_settings.json 2>/dev/null || true
fi
if [ -f /root/.openhands/mcp.json ]; then
    cp /root/.openhands/mcp.json ~/.openhands/mcp.json 2>/dev/null || true
fi
# Verify config is readable
if [ -f ~/.openhands/agent_settings.json ]; then
    echo "Config found at ~/.openhands/agent_settings.json"
    cat ~/.openhands/agent_settings.json | python3 -c 'import sys,json; json.load(sys.stdin)' 2>&1 && echo "Config is valid JSON" || echo "Config is INVALID JSON"
else
    echo "WARNING: Config not found at ~/.openhands/agent_settings.json"
fi
'''
        
        full_cmd = f"{pre_cmd}\n{oh_cmd}"
        
        return [
            "docker", "exec", "-it",
            "-e", f"PATH={CONTAINER_PATH}",
            "-e", "RUNTIME=local",
            "-e", "HOME=/root",
            name,
            "bash", "-c", full_cmd
        ]
    
    @staticmethod
    def start_background_session(
        project: Project,
        resume_session: Optional[str] = None,
        task: Optional[str] = None
    ) -> bool:
        """Start OpenHands in background via tmux inside container."""
        name = project.container_name
        
        if not Docker.container_running(name):
            return False
        
        if resume_session:
            oh_cmd = f'openhands --always-approve --resume "{resume_session}"'
        elif task:
            # Write task to file (too large for command line)
            task_file = "/tmp/openhands_task.txt"
            write_cmd = f"cat > {task_file} << 'TASKEOF'\n{task}\nTASKEOF"
            Docker.exec_in_container(name, write_cmd, timeout=10)
            oh_cmd = f"openhands --always-approve -t \"$(cat {task_file})\""
        else:
            oh_cmd = 'openhands --always-approve'
        
        # Ensure config is readable before starting (same as interactive mode)
        pre_cmd = f'''
export PATH="{CONTAINER_PATH}"
export RUNTIME=local
export HOME=/root
# Ensure .openhands directory and config exist and are readable
mkdir -p ~/.openhands
if [ -f /root/.openhands/agent_settings.json ]; then
    cp /root/.openhands/agent_settings.json ~/.openhands/agent_settings.json 2>/dev/null || true
    chmod 644 ~/.openhands/agent_settings.json 2>/dev/null || true
fi
if [ -f /root/.openhands/mcp.json ]; then
    cp /root/.openhands/mcp.json ~/.openhands/mcp.json 2>/dev/null || true
fi
'''
        
        tmux_script = f'''
{pre_cmd}
tmux kill-session -t {TMUX_OH_SESSION} 2>/dev/null || true
tmux new-session -d -s {TMUX_OH_SESSION} '{oh_cmd}'
'''
        
        code, output = Docker.exec_in_container(name, tmux_script, timeout=30)
        return code == 0
    
    @staticmethod
    def is_background_running(project: Project) -> bool:
        """Check if background OpenHands session is running."""
        name = project.container_name
        
        if not Docker.container_running(name):
            return False
        
        code, output = Docker.exec_in_container(
            name,
            f"tmux has-session -t {TMUX_OH_SESSION} 2>/dev/null && echo 'yes'",
            timeout=5
        )
        return "yes" in output
    
    @staticmethod
    def attach_background_session(project: Project) -> List[str]:
        """Get command to attach to background session.
        
        Uses 'tmux attach' which shares the terminal with the running session.
        Clipboard works via terminal emulator (kitty, etc).
        Detach with Ctrl+B, D to leave session running.
        """
        return [
            "docker", "exec", "-it",
            "-e", f"PATH={CONTAINER_PATH}",
            "-e", "TERM=xterm-256color",
            project.container_name,
            "tmux", "attach-session", "-t", TMUX_OH_SESSION
        ]
    
    @staticmethod
    def stop_background_session(project: Project) -> bool:
        """Stop background OpenHands session."""
        name = project.container_name
        
        if not Docker.container_running(name):
            return True
        
        Docker.exec_in_container(
            name,
            f"tmux kill-session -t {TMUX_OH_SESSION} 2>/dev/null || true",
            timeout=10
        )
        return True
    
    @staticmethod
    def get_background_output(project: Project, lines: int = 30) -> str:
        """Get recent output from background session."""
        name = project.container_name
        
        if not Docker.container_running(name):
            return ""
        
        code, output = Docker.exec_in_container(
            name,
            f"tmux capture-pane -t {TMUX_OH_SESSION} -p 2>/dev/null | tail -{lines}",
            timeout=10
        )
        return output if code == 0 else ""


# =============================================================================
# PROJECT MANAGER
# =============================================================================

class ProjectManager:
    """Manage OpenHands projects with enhanced edge case handling."""
    
    @staticmethod
    def init_directories():
        """Initialize directory structure."""
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        LLM_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        MCP_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        RALPH_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        SKILLS_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def list_projects() -> List[Project]:
        """List all projects."""
        projects = []
        if PROJECTS_DIR.exists():
            for p in sorted(PROJECTS_DIR.iterdir()):
                if p.is_dir() and (p / "workspace").exists():
                    projects.append(Project(name=p.name, path=p))
        return projects
    
    @staticmethod
    def get_project(name: str) -> Optional[Project]:
        """Get project by name."""
        path = PROJECTS_DIR / name
        if path.exists() and (path / "workspace").exists():
            return Project(name=name, path=path)
        return None
    
    @staticmethod
    def create_project(name: str, llm_template: Optional[str] = None, pull_image: bool = True) -> Optional[Project]:
        """Create new project."""
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$', name):
            log_error(f"Invalid project name: {name}")
            return None
        
        path = PROJECTS_DIR / name
        if path.exists():
            log_error(f"Project already exists: {name}")
            return None
        
        try:
            if pull_image:
                log(f"Pulling latest image: {RUNTIME_IMAGE}")
                Docker.pull_image()
            
            (path / "workspace").mkdir(parents=True)
            openhands_dir = path / "config" / ".openhands"
            openhands_dir.mkdir(parents=True, exist_ok=True)
            
            if llm_template:
                llm_path = LLM_TEMPLATES_DIR / llm_template / "agent_settings.json"
                if llm_path.exists():
                    dest = openhands_dir / "agent_settings.json"
                    shutil.copy(llm_path, dest)
                    # Set permissions so root in container can read it
                    os.chmod(dest, 0o644)
                    log(f"Copied agent_settings.json from template: {llm_template}")
                else:
                    log_error(f"LLM template not found: {llm_template}")
            else:
                log("WARNING: No LLM template selected. Put agent_settings.json manually to config/.openhands/")
            
            # Verify agent_settings.json exists and is readable
            agent_settings = openhands_dir / "agent_settings.json"
            if agent_settings.exists():
                try:
                    with open(agent_settings, encoding='utf-8') as f:
                        config = json.load(f)
                        model = config.get('llm', {}).get('model', 'UNKNOWN')
                        log(f"Project config: model={model}")
                except Exception as e:
                    log_error(f"Invalid agent_settings.json: {e}")
            else:
                log("WARNING: agent_settings.json not found! OpenHands will show setup wizard.")
            
            mcp_path = MCP_TEMPLATES_DIR / "mcp_servers.json"
            if mcp_path.exists():
                shutil.copy(mcp_path, openhands_dir / "mcp_servers.json")
            
            if SKILLS_TEMPLATES_DIR.exists():
                skills_dest = openhands_dir / "skills"
                skills_dest.mkdir(exist_ok=True)
                for skill_dir in SKILLS_TEMPLATES_DIR.iterdir():
                    if skill_dir.is_dir():
                        shutil.copytree(skill_dir, skills_dest / skill_dir.name, dirs_exist_ok=True)
            
            (path / ".persistent").touch()
            
            log(f"Created project: {name}")
            return Project(name=name, path=path)
            
        except Exception as e:
            log_error(f"Failed to create project {name}: {e}")
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            return None
    
    @staticmethod
    def delete_project(name: str, remove_container: bool = True) -> bool:
        """Delete project and optionally its container with proper cleanup."""
        project = ProjectManager.get_project(name)
        if not project:
            return False
        
        try:
            # Stop Ralph daemon in container
            try:
                if Docker.container_running(project.container_name):
                    Docker.stop_ralph_daemon(project)
                    log(f"Stopped Ralph daemon in container")
            except Exception:
                pass
            
            # Stop background OpenHands session
            try:
                Docker.stop_background_session(project)
            except Exception:
                pass
            
            # Fix permissions BEFORE removing container
            # Files created in container are owned by root, delete them via docker exec
            if Docker.container_exists(project.container_name):
                try:
                    log("Cleaning files via container...")
                    
                    # Start container if not running (needed for exec)
                    container_was_running = Docker.container_running(project.container_name)
                    if not container_was_running:
                        log("Starting container for cleanup...")
                        Docker.start_container(project.container_name)
                        time.sleep(1)
                    
                    # Delete files from inside container (as root)
                    Docker.exec_in_container(
                        project.container_name,
                        "rm -rf /workspace/* /root/.openhands /root/.cache /root/.local 2>/dev/null; true",
                        timeout=30
                    )
                    
                    # If we started it just for cleanup, stop it
                    if not container_was_running:
                        Docker.stop_container(project.container_name)
                        
                except Exception as e:
                    log(f"Could not clean via container: {e}")
            
            # Stop and remove container
            if remove_container:
                Docker.stop_container(project.container_name)
                Docker.remove_container(project.container_name, force=True)
            
            time.sleep(0.5)
            
            # Remove files (should work now since container cleaned them)
            try:
                shutil.rmtree(project.path)
            except PermissionError:
                log(f"Permission denied, trying force deletion for: {name}")
                
                # Fallback: use docker run with alpine to cleanup as root
                try:
                    cleanup_cmd = [
                        "docker", "run", "--rm",
                        "-v", f"{project.path}:/target",
                        "alpine:latest",
                        "sh", "-c", "rm -rf /target/* /target/.* 2>/dev/null; true"
                    ]
                    subprocess.run(cleanup_cmd, capture_output=True, timeout=30)
                except Exception as e:
                    log(f"Docker cleanup fallback failed: {e}")
                
                # Try with elevated permissions if available
                for cmd in [
                    ["rm", "-rf", str(project.path)],
                    ["sudo", "-n", "rm", "-rf", str(project.path)],  # -n = non-interactive
                ]:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if not project.path.exists():
                        break
                
                # Last resort - ignore errors
                if project.path.exists():
                    shutil.rmtree(project.path, ignore_errors=True)
                    if project.path.exists():
                        log_error(f"Could not fully delete {name}, manual cleanup may be needed: {project.path}")
                        # Don't fail - return True anyway, user can clean up manually
            
            log(f"Deleted project: {name}")
            return True
            
        except Exception as e:
            log_error(f"Failed to delete project {name}: {e}")
            return False
    
    @staticmethod
    def list_llm_templates() -> List[str]:
        """List available LLM templates."""
        templates = []
        if LLM_TEMPLATES_DIR.exists():
            for d in sorted(LLM_TEMPLATES_DIR.iterdir()):
                if d.is_dir() and (d / "agent_settings.json").exists():
                    templates.append(d.name)
        return templates
    
    @staticmethod
    def get_sessions(project: Project) -> List[Dict]:
        """Get list of sessions for project, sorted by modification time (newest first)."""
        sessions = []
        conv_dir = project.openhands_dir / "conversations"
        if conv_dir.exists():
            for sid in conv_dir.iterdir():
                if sid.is_dir():
                    try:
                        mtime = sid.stat().st_mtime
                        age = datetime.now() - datetime.fromtimestamp(mtime)
                        sessions.append({
                            "id": sid.name,
                            "path": sid,
                            "mtime": mtime,
                            "age_days": age.days
                        })
                    except Exception:
                        pass
        # Sort by mtime descending (newest first)
        sessions.sort(key=lambda x: x["mtime"], reverse=True)
        return sessions[:50]
    
    @staticmethod
    def change_llm(project: Project, template: str) -> bool:
        """Change LLM template for project."""
        try:
            llm_path = LLM_TEMPLATES_DIR / template / "agent_settings.json"
            if not llm_path.exists():
                log_error(f"Template not found: {template}")
                return False
            
            dest = project.openhands_dir / "agent_settings.json"
            shutil.copy(llm_path, dest)
            log(f"Changed LLM for {project.name} to {template}")
            return True
        except Exception as e:
            log_error(f"Failed to change LLM: {e}")
            return False
    
    @staticmethod
    def update_mcp(project: Project) -> bool:
        """Update MCP config from templates and clear cache."""
        try:
            mcp_path = MCP_TEMPLATES_DIR / "mcp_servers.json"
            if not mcp_path.exists():
                log_error("MCP template not found")
                return False
            
            dest = project.openhands_dir / "mcp_servers.json"
            shutil.copy(mcp_path, dest)
            
            # Clear MCP cache markers
            for marker in [".mcp_hash", ".mcp_packages", ".mcp_last_update", 
                          ".mcp_warmup_done", ".gateway_setup_done", ".gateway_pids",
                          "mcp.json"]:
                marker_path = project.openhands_dir / marker
                if marker_path.exists():
                    marker_path.unlink()
            
            log(f"Updated MCP for {project.name}")
            return True
        except Exception as e:
            log_error(f"Failed to update MCP: {e}")
            return False
    
    @staticmethod
    def update_skills(project: Project) -> bool:
        """Update skills from templates."""
        try:
            if not SKILLS_TEMPLATES_DIR.exists():
                log_error("Skills template dir not found")
                return False
            
            skills_dest = project.openhands_dir / "skills"
            
            if skills_dest.exists():
                shutil.rmtree(skills_dest)
            
            # Check if QMD is enabled
            qmd_enabled = (project.workspace / ".qmd_enabled").exists()
            
            skills_dest.mkdir(exist_ok=True)
            for skill_dir in SKILLS_TEMPLATES_DIR.iterdir():
                if skill_dir.is_dir():
                    # Skip qmd-search skill if QMD not enabled
                    if skill_dir.name == "qmd-search" and not qmd_enabled:
                        continue
                    shutil.copytree(skill_dir, skills_dest / skill_dir.name, dirs_exist_ok=True)
            
            log(f"Updated skills for {project.name}")
            return True
        except Exception as e:
            log_error(f"Failed to update skills: {e}")
            return False
    
    @staticmethod
    def reset_container(project: Project) -> bool:
        """Full reset - container + Ralph data. Keeps workspace files only."""
        try:
            name = project.container_name
            
            # Stop Ralph daemon in container
            try:
                if Docker.container_running(name):
                    Docker.stop_ralph_daemon(project)
                    log(f"Stopped Ralph daemon")
            except Exception:
                pass
            
            # Remove container
            Docker.stop_container(name)
            Docker.remove_container(name, force=True)
            log(f"Removed container {name}")
            
            # Delete .ralph folder
            if project.ralph_dir.exists():
                shutil.rmtree(project.ralph_dir)
                log(f"Deleted Ralph data")
            
            # Clear setup markers
            markers = [".setup_done", ".mcp_warmup_done", ".mcp_hash", ".mcp_packages", ".mcp_last_update"]
            for m in markers:
                mp = project.openhands_dir / m
                if mp.exists():
                    mp.unlink()
            
            gateway_pids = project.openhands_dir / ".gateway_pids"
            if gateway_pids.exists():
                gateway_pids.unlink()
            
            # Create fresh container
            print("* Creating new container...")
            if not Docker.create_persistent_container(project):
                log_error("Failed to create new container")
                return False
            
            log(f"Full reset completed for {project.name}")
            return True
        except Exception as e:
            log_error(f"Failed to reset: {e}")
            return False



# =============================================================================
# RALPH MANAGER - Enhanced with edge case handling
# =============================================================================

class RalphManager:
    """Manage Ralph autonomous coding loop with state preservation and self-healing."""
    
    # Container path prefix for Docker operations
    CONTAINER_RALPH_DIR = "/workspace/.ralph"
    
    def __init__(self, project: Project):
        self.project = project
        self.container_name = project.container_name
        
        # Both point to workspace/.ralph/ - single directory for all Ralph state
        # This directory is git-tracked and contains both runtime state and project files
        self.ralph_dir = project.ralph_dir  # workspace/.ralph/
        self.workspace_ralph = project.ralph_dir  # Same as ralph_dir
        
        # Ensure workspace .ralph directory exists (via Docker if container running)
        self._ensure_ralph_dir_exists()
        
        # === GIT-TRACKED FILES (in workspace/.ralph/) ===
        # These are committed to git and shared across workers
        self.prd_file = self.workspace_ralph / "prd.json"          # Tasks
        self.mission_file = self.workspace_ralph / "MISSION.md"    # Project goal
        self.learnings_file = self.workspace_ralph / "LEARNINGS.md"  # Accumulated knowledge
        self.architecture_file = self.workspace_ralph / "ARCHITECTURE.md"  # Architecture
        self.progress_file = self.workspace_ralph / "PROGRESS.md"  # Milestone progress
        self.status_file = self.workspace_ralph / "STATUS.md"      # Current status
        
        # === RUNTIME FILES (in ralph_dir) ===
        # These are NOT in git - internal Ralph state
        self.config_file = self.ralph_dir / "config.json"        # Runtime config
        self.checkpoint_file = self.ralph_dir / "checkpoint.json"  # Recovery
        self.runtime_log = self.ralph_dir / "progress.txt"        # Detailed log (legacy)
        
        # Pre-create all files so daemon preserves ownership when updating
        self._ensure_ralph_files_exist()
        
        # Thread-safe stop flag
        self._stop_event = threading.Event()
        
        # Knowledge and context managers
        self.knowledge = KnowledgeManager(self.ralph_dir, self.container_name)
        self.context_manager = ContextWindowManager()
        self.state_manager = StateManager(self.ralph_dir, self.workspace_ralph)
        self.disk_monitor = DiskSpaceMonitor(self.ralph_dir)
        
        # Autonomous operation components
        self.git_gate = GitQualityGate(project.workspace)
        self.git_handoff = GitHandoff(project.workspace, self.ralph_dir)
        # Pass container name so tests run INSIDE container (where cargo/npm/go are available)
        self.test_enforcer = TestEnforcement(project.workspace, self.ralph_dir, project.container_name)
        self.stuck_detector = StuckDetector(self.ralph_dir)
        self.memory = HierarchicalMemory(self.ralph_dir)
        self.condenser = ContextCondenser(self.ralph_dir)
        
        # NEW: Semantic search, metrics, and learnings manager
        self.semantic = SemanticSearch(self.ralph_dir)
        self.metrics = RalphMetrics(self.ralph_dir)
        self.learnings_mgr = LearningsManager(self.ralph_dir, self.semantic)
        
        # Health monitoring for resilience
        self.health_monitor = HealthMonitor(project)
        
        # Notification system for alerting
        self.notifier = NotificationManager(self.ralph_dir)
    
    def _ensure_ralph_dir_exists(self):
        """Ensure .ralph directory and all subdirectories exist via Docker."""
        if not Docker.container_running(self.container_name):
            Docker.ensure_container_running(self.project)
        if not Docker.container_running(self.container_name):
            return  # Failed to start container
        
        # Create all directories that Ralph and helper classes need
        dirs_to_create = [
            self.CONTAINER_RALPH_DIR,
            f"{self.CONTAINER_RALPH_DIR}/iterations",
            f"{self.CONTAINER_RALPH_DIR}/prompts",
            f"{self.CONTAINER_RALPH_DIR}/backups",
            f"{self.CONTAINER_RALPH_DIR}/epochs",
            f"{self.CONTAINER_RALPH_DIR}/knowledge",  # KnowledgeManager
            f"{self.CONTAINER_RALPH_DIR}/memory",     # HierarchicalMemory
            f"{self.CONTAINER_RALPH_DIR}/state",      # StateManager
        ]
        for d in dirs_to_create:
            Docker.mkdir(self.container_name, d)
    
    def _ensure_ralph_files_exist(self):
        """
        Create all workspace/.ralph/ files via Docker.
        
        All file operations go through Docker to avoid permission issues.
        """
        if not Docker.container_running(self.container_name):
            Docker.ensure_container_running(self.project)
        if not Docker.container_running(self.container_name):
            return  # Failed to start container
        
        # Create subdirectories
        for subdir in ["iterations", "prompts", "backups", "epochs"]:
            Docker.mkdir(self.container_name, f"{self.CONTAINER_RALPH_DIR}/{subdir}")
        
        # Files to create with initial content
        files_to_create = {
            "config.json": "{}",
            "prd.json": "{}",
            "MISSION.md": "# Mission\n\n",
            "LEARNINGS.md": "# Learnings\n\n",
            "ARCHITECTURE.md": "# Architecture\n\n",
            "PROGRESS.md": "# Progress\n\n",
            "STATUS.md": "# Status\n\n",
            "checkpoint.json": "{}",
            "heartbeat": "",
            "ralph.log": "",
            "condensed_context.md": "",
            "condense_history.json": "[]",
            "stuck_history.json": "[]",
            "test_history.json": "[]",
            "metrics.json": "{}",
            "memory.json": "{}",
            "semantic_cache.json": "{}",
            "handoff.json": "{}",
            "task_context.json": "{}",
            "task_archive.json": "[]",
        }
        
        for filename, default_content in files_to_create.items():
            path = f"{self.CONTAINER_RALPH_DIR}/{filename}"
            if not Docker.file_exists(self.container_name, path):
                Docker.write_file(self.container_name, path, default_content)
    
    # =========================================================================
    # DOCKER FILE OPERATION HELPERS
    # All .ralph file operations use these to avoid permission issues
    # =========================================================================
    
    def _ensure_container(self) -> bool:
        """Ensure container is running, start if needed."""
        if Docker.container_running(self.container_name):
            return True
        return Docker.ensure_container_running(self.project)
    
    def _container_path(self, filename: str) -> str:
        """Get container path for a ralph file."""
        return f"{self.CONTAINER_RALPH_DIR}/{filename}"
    
    def _read_file(self, filename: str) -> Optional[str]:
        """Read file from container."""
        if not self._ensure_container():
            return None
        return Docker.read_file(self.container_name, self._container_path(filename))
    
    def _write_file(self, filename: str, content: str) -> bool:
        """Write file to container."""
        if not self._ensure_container():
            return False
        return Docker.write_file(self.container_name, self._container_path(filename), content)
    
    def _read_json(self, filename: str, default: Any = None) -> Any:
        """Read JSON file from container."""
        if not self._ensure_container():
            return default
        return Docker.read_json(self.container_name, self._container_path(filename), default)
    
    def _write_json(self, filename: str, data: Any) -> bool:
        """Write JSON file to container."""
        if not self._ensure_container():
            return False
        return Docker.write_json(self.container_name, self._container_path(filename), data)
    
    def _append_file(self, filename: str, content: str) -> bool:
        """Append to file in container."""
        if not self._ensure_container():
            return False
        return Docker.append_file(self.container_name, self._container_path(filename), content)
    
    def _file_exists(self, filename: str) -> bool:
        """Check if file exists in container."""
        if not self._ensure_container():
            return False
        return Docker.file_exists(self.container_name, self._container_path(filename))
    
    def _init_knowledge_files(self):
        """Initialize knowledge structure files via Docker."""
        knowledge_files = {
            "knowledge/critical.md": """# Critical Knowledge

> This file contains CRITICAL information that MUST NOT be lost.
> Only add information that is essential for project success.

## Project Architecture

(Added by Architect iterations)

## Critical Patterns

(Patterns that must be followed)

## Critical Issues to Avoid

(Issues that caused major problems)

## External API Details

(API endpoints, authentication, etc.)

## Build/Deploy Requirements

(Essential build steps)
""",
            "knowledge/patterns.md": """# Code Patterns

## Successful Patterns

(Patterns that worked well)

## Anti-Patterns

(Patterns to avoid)

## Refactoring Patterns

(How to safely refactor)
""",
            "knowledge/issues.md": """# Known Issues

## Resolved Issues

(Issues that were fixed - include solution)

## Open Issues

(Issues still being worked on)

## Workarounds

(Temporary solutions)
""",
            "knowledge/apis.md": """# External APIs & Services

## Service: [Name]

- Endpoint: 
- Auth: 
- Rate limits: 
- Key methods: 
""",
            "knowledge/decisions.md": """# Architecture Decisions

## Decision: [Title]

- Date: 
- Context: 
- Decision: 
- Consequences: 
"""
        }
        
        for filename, content in knowledge_files.items():
            path = f"{self.CONTAINER_RALPH_DIR}/{filename}"
            if not Docker.file_exists(self.container_name, path):
                Docker.write_file(self.container_name, path, content)
    
    def scan_learnings_for_red_flags(self, current_task_id: Optional[str] = None) -> Tuple[bool, str, float]:
        """
        Scan LEARNINGS.md for red flags with weighted scoring.
        
        Returns: (has_blocking_flag, reason, max_weight)
        - has_blocking_flag: True if found red flag with weight > 0.5
        - reason: Description of the red flag
        - max_weight: Highest weight of found red flags
        
        Weight calculation:
        - 1.0: In current task section (## TASK-XXX)
        - 0.9: In last 10% of file (most recent)
        - 0.7: In last 30% of file
        - 0.5: In last 50% of file  
        - 0.3: Older content
        
        Red flags with weight >= 0.5 block verification.
        """
        content = self._read_file("LEARNINGS.md")
        if not content:
            return False, "", 0.0
        
        try:
            content_lower = content.lower()
            content_len = len(content)
            
            if content_len == 0:
                return False, "", 0.0
            
            # Red flags that indicate goal was NOT achieved
            red_flags = [
                "no orders",
                "not placed",
                "0 orders",
                "zero orders",
                "not trading",
                "not working",
                "failed to",
                "could not",
                "unable to",
                "dry-run",
                "dry run",
                "allowance is $0",
                "allowance: 0",
                "allowance.*0",
                "all signals are neutral",
                "signals are neutral",
                "goal not achieved",
                "not executed",
                "no trades",
                "не торгует",
                "не работает",
                "ордеров нет",
                "нет ордеров",
            ]
            
            found_flags = []  # List of (flag, position, weight, in_current_task)
            
            # Parse sections by ## headers
            sections = []
            current_pos = 0
            for match in re.finditer(r'^## (.+)$', content, re.MULTILINE):
                if current_pos > 0:
                    sections.append((sections[-1][0] if sections else "", current_pos, match.start()))
                sections.append((match.group(1), match.start(), None))
                current_pos = match.start()
            if sections and sections[-1][2] is None:
                sections[-1] = (sections[-1][0], sections[-1][1], content_len)
            
            # Find current task section if specified
            current_task_section = None
            if current_task_id:
                for section_name, start, end in sections:
                    if current_task_id.lower() in section_name.lower():
                        current_task_section = (start, end)
                        break
            
            # Scan for red flags
            for flag in red_flags:
                # Use regex for pattern matching
                if ".*" in flag:
                    pattern = flag.replace(".*", ".*?")
                    matches = list(re.finditer(pattern, content_lower))
                else:
                    # Find all occurrences
                    matches = []
                    start = 0
                    while True:
                        pos = content_lower.find(flag, start)
                        if pos == -1:
                            break
                        matches.append(type('Match', (), {'start': lambda s=pos: s, 'group': lambda s=flag: s})())
                        start = pos + 1
                
                for match in matches:
                    pos = match.start()
                    
                    # Calculate weight based on position
                    relative_pos = pos / content_len
                    
                    # Check if in current task section (highest priority)
                    in_current_task = False
                    if current_task_section:
                        start, end = current_task_section
                        if start <= pos < end:
                            in_current_task = True
                    
                    if in_current_task:
                        weight = 1.0
                    elif relative_pos >= 0.9:  # Last 10%
                        weight = 0.9
                    elif relative_pos >= 0.7:  # Last 30%
                        weight = 0.7
                    elif relative_pos >= 0.5:  # Last 50%
                        weight = 0.5
                    else:
                        weight = 0.3
                    
                    found_flags.append((flag, pos, weight, in_current_task))
            
            if not found_flags:
                return False, "", 0.0
            
            # Find the most severe flag (highest weight)
            found_flags.sort(key=lambda x: x[2], reverse=True)
            top_flag, top_pos, top_weight, in_task = found_flags[0]
            
            # Extract context around the flag
            context_start = max(0, top_pos - 50)
            context_end = min(content_len, top_pos + len(top_flag) + 100)
            context = content[context_start:context_end].replace('\n', ' ').strip()
            
            location = "in current task" if in_task else f"at {int(top_pos/content_len*100)}% of learnings"
            reason = f"Found '{top_flag}' {location}: ...{context}..."
            
            # Block if weight >= 0.5
            should_block = top_weight >= 0.5
            
            return should_block, reason, top_weight
            
        except Exception as e:
            print(f"[scan_learnings] Error: {e}")
            return False, "", 0.0
    
    def init_structure(self, config: RalphConfig) -> bool:
        """Initialize Ralph structure for project via Docker.
        
        CRITICAL FIX: Uses .init_in_progress marker to detect interrupted initialization
        and prevent state corruption after container crashes."""
        marker_path = f"{self.CONTAINER_RALPH_DIR}/.init_in_progress"
        
        try:
            # Ensure container is running
            if not self._ensure_container():
                log_error("Failed to start container for Ralph init")
                return False
            
            # Create marker to detect interrupted init
            Docker.write_file(self.container_name, marker_path, datetime.now().isoformat())
            
            # === CREATE ALL DIRECTORIES VIA DOCKER ===
            dirs_to_create = [
                self.CONTAINER_RALPH_DIR,
                f"{self.CONTAINER_RALPH_DIR}/logs",
                f"{self.CONTAINER_RALPH_DIR}/verification",
                f"{self.CONTAINER_RALPH_DIR}/checkpoints",
                f"{self.CONTAINER_RALPH_DIR}/iterations",
                f"{self.CONTAINER_RALPH_DIR}/.tmp",
                f"{self.CONTAINER_RALPH_DIR}/backups",
                f"{self.CONTAINER_RALPH_DIR}/knowledge",
                f"{self.CONTAINER_RALPH_DIR}/memory",
                f"{self.CONTAINER_RALPH_DIR}/prompts",
                f"{self.CONTAINER_RALPH_DIR}/epochs",
                f"{self.CONTAINER_RALPH_DIR}/condense_backups",
                f"{self.CONTAINER_RALPH_DIR}/iterations_archive",
                f"{self.CONTAINER_RALPH_DIR}/iterations_backup",
            ]
            for d in dirs_to_create:
                Docker.mkdir(self.container_name, d)
            
            # Initialize knowledge files via Docker
            self._init_knowledge_files()
            
            # Initialize git in workspace if not already
            if not (self.project.workspace / ".git").exists():
                subprocess.run(
                    ['git', 'init'],
                    capture_output=True, cwd=self.project.workspace, timeout=10
                )
                subprocess.run(
                    ['git', 'config', 'user.email', 'ralph@autonomous.dev'],
                    capture_output=True, cwd=self.project.workspace, timeout=5
                )
                subprocess.run(
                    ['git', 'config', 'user.name', 'Ralph'],
                    capture_output=True, cwd=self.project.workspace, timeout=5
                )
            
            # Create config.json (runtime state - not in git)
            config_data = {
                "maxIterations": config.max_iterations,
                "sessionTimeoutSeconds": config.session_timeout,
                "architectInterval": config.architect_interval,
                "condenseInterval": config.condense_interval,
                "condenseBeforeArchitect": config.condense_before_architect,
                "requireVerification": config.require_verification,
                "pauseBetweenSeconds": config.pause_between,
                "currentIteration": 0,
                "status": "initialized",
                "pauseRequested": False,
                "stuckCount": 0,
                "createdAt": datetime.now().isoformat(),
                "taskDescription": config.task_description,
                "version": VERSION
            }
            self._write_json("config.json", config_data)
            
            # === GIT-TRACKED FILES (in workspace/.ralph/) ===
            
            # Create prd.json (git-tracked - tasks and state)
            prd_data = {
                "projectName": self.project.name,
                "phase": "planning",
                "verified": False,
                "taskDescription": config.task_description,
                "userStories": [{
                    "id": "PLAN",
                    "title": "Analyze and create execution plan",
                    "description": "Read the codebase, understand requirements, create detailed plan",
                    "type": "planning",
                    "priority": 0,
                    "passes": False,
                    "dependsOn": []
                }]
            }
            self._write_json("prd.json", prd_data)
            
            # Create MISSION.md (git-tracked)
            mission_content = f"""# MISSION

> This file is READ-ONLY. It captures the original request.
> Every iteration reads this to stay focused on the actual goal.

## Original Request

{config.task_description}

## Success Criteria

The project is DONE when:
1. All requirements above are implemented
2. Code builds without errors
3. Tests pass
4. The application works as described

---

Created: {datetime.now().isoformat()}
DO NOT MODIFY THIS FILE
"""
            self._write_file("MISSION.md", mission_content)
            
            # Create ARCHITECTURE.md (git-tracked)
            self._write_file("ARCHITECTURE.md", """# Architecture Decisions

Updated by Architect iterations (every N iterations).

## Project Structure

(To be filled by Architect review)

## Key Decisions

(Important architectural decisions)

## Patterns

(Code patterns to follow)

## Known Issues

(Problems found during reviews)
""")
            
            # Create LEARNINGS.md (git-tracked)
            self._write_file("LEARNINGS.md", f"""# Ralph Learnings

> This file accumulates knowledge during development.
> READ THIS at the start of each iteration to avoid repeating mistakes.
> UPDATE THIS when you discover something important.

## Project: {self.project.name}
## Created: {datetime.now().isoformat()}

---

## Key Discoveries

(Add important findings here)

## What Works Well

(Patterns and approaches that succeeded)

## What To Avoid

(Things that failed or caused problems)

---

Update this file after significant discoveries or failures
""")
            
            # Create PROGRESS.md (git-tracked - milestone progress)
            self._write_file("PROGRESS.md", f"""# Progress Log

**Project:** {self.project.name}
**Started:** {datetime.now().isoformat()}

## Milestones

""")
            
            # Create STATUS.md (git-tracked)
            self._write_file("STATUS.md", f"""# Project Status

**Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status:** initialized

## Progress

- Tasks completed: 0/1
- Completion: 0%

## Tasks

- ⏳ PLAN: Analyze and create execution plan
""")
            
            # Initial git commit for .ralph/
            subprocess.run(
                ['git', 'add', '.ralph/'],
                capture_output=True, cwd=self.project.workspace, timeout=10
            )
            subprocess.run(
                ['git', 'commit', '-m', '[Ralph] Initialize project structure'],
                capture_output=True, cwd=self.project.workspace, timeout=30
            )
            
            # Enable QMD by default for Ralph projects
            # QMD provides semantic code search which helps the model find relevant code
            qmd_mgr = QMDManager(self.project)
            if not qmd_mgr.is_enabled:
                qmd_mgr.enable()
                log(f"QMD semantic search enabled by default")
            
            # Create AGENTS.md
            self._write_file("AGENTS.md", """# Project Operations Guide

## Build & Run
- Build: `npm run build` or `python -m build` or `go build ./...`
- Dev: `npm run dev` or `python app.py` or `go run main.go`

## Validation
- Tests: `npm test` or `pytest` or `go test ./...`
- Typecheck: `npm run typecheck` or `mypy .`
- Lint: `npm run lint` or `ruff check .`

## Codebase Patterns
(Add discovered patterns here)

## Operational Notes
(Add learned commands/quirks here)
""")
            
            # Create guardrails.md (runtime - for internal tracking)
            self._write_file("guardrails.md", """# Guardrails

When something fails, add a SIGN here so future iterations avoid the mistake.

## Signs

(Add signs for failures here)
""")
            
            # Create runtime progress.txt (detailed log - not git)
            self._write_file("progress.txt", f"# Detailed Progress Log\n# Started: {datetime.now().isoformat()}\n\n")
            
            # CRITICAL FIX: Remove init_in_progress marker after ALL files created
            # This prevents state corruption if container crashes during init
            marker_path = f"{self.CONTAINER_RALPH_DIR}/.init_in_progress"
            Docker.delete_file(self.container_name, marker_path)
            
            log(f"Initialized Ralph structure for: {self.project.name}")
            return True
            
        except Exception as e:
            log_error(f"Failed to init Ralph: {e}")
            log_error(traceback.format_exc())
            return False
    
    def get_config(self) -> dict:
        """Get current Ralph config via Docker."""
        return self._read_json("config.json", default={})
    
    def is_initialized(self) -> bool:
        """Check if Ralph structure is fully initialized (no init_in_progress marker).
        
        CRITICAL FIX: Checks for .init_in_progress marker to detect interrupted
        initialization (e.g., container crash during setup). If marker exists,
        initialization did not complete and should be re-run.
        """
        # Check if marker exists - if so, init was interrupted
        marker_path = f"{self.CONTAINER_RALPH_DIR}/.init_in_progress"
        if Docker.file_exists(self.container_name, marker_path):
            return False  # Init was interrupted
        
        # Check for config.json as indicator of successful init
        config_path = f"{self.CONTAINER_RALPH_DIR}/config.json"
        return Docker.file_exists(self.container_name, config_path)
    
    def update_config(self, key: str, value: Any) -> bool:
        """Update config value via Docker."""
        try:
            config = self._read_json("config.json", default={})
            config[key] = value
            return self._write_json("config.json", config)
        except Exception as e:
            log(f"Error updating config: {e}")
            return False
    
    def get_prd(self) -> dict:
        """Get current PRD via Docker with corruption recovery."""
        default_prd = {
            "projectName": self.project.name,
            "phase": "planning",
            "verified": False,
            "userStories": []
        }
        
        prd = self._read_json("prd.json", default=None)
        
        if prd is None:
            return default_prd
        
        # Sanitize: recover tasks with invalid IDs
        valid_id_pattern = re.compile(r'^[A-Za-z0-9_\-]+$')
        stories = prd.get("userStories", [])
        needs_save = False
        
        for idx, story in enumerate(stories):
            task_id = story.get("id", "")
            if not task_id or not valid_id_pattern.match(task_id) or len(task_id) >= 100:
                recovered_id = f"RECOVERED-{int(time.time())}-{idx}"
                story["_corrupted_id"] = task_id
                story["id"] = recovered_id
                log(f"Recovered task with invalid ID: {repr(task_id)[:50]} → {recovered_id}")
                needs_save = True
        
        if needs_save:
            self._write_json("prd.json", prd)
        
        return prd
    
    def save_prd(self, prd: dict, commit: bool = False, message: str = None) -> bool:
        """Save PRD via Docker.
        
        Args:
            prd: PRD data to save
            commit: If True, commit changes to git
            message: Commit message (optional)
        """
        try:
            result = self._write_json("prd.json", prd)
            
            if result and commit:
                self.commit_workspace_ralph(message or "Update task status")
            
            return result
        except Exception as e:
            log_error(f"Failed to save PRD: {e}")
            return False
    
    def save_checkpoint(self, iteration: int, task_id: str, status: str = "in_progress"):
        """Save checkpoint via Docker."""
        checkpoint = {
            "iteration": iteration,
            "task_id": task_id,
            "prd_state": self.get_prd(),
            "config_state": self.get_config(),
            "timestamp": datetime.now().isoformat(),
            "status": status
        }
        if not self._write_json("checkpoint.json", checkpoint):
            log_error(f"Failed to save checkpoint for iteration {iteration}")
    
    def load_checkpoint(self) -> Optional[Checkpoint]:
        """Load last checkpoint via Docker."""
        data = self._read_json("checkpoint.json", default=None)
        if data is None:
            return None
        
        try:
            return Checkpoint(
                iteration=data.get("iteration", 0),
                task_id=data.get("task_id", ""),
                prd_state=data.get("prd_state", {}),
                timestamp=data.get("timestamp", ""),
                status=data.get("status", "unknown")
            )
        except Exception:
            return None
    
    def cleanup_old_data(self):
        """Periodic cleanup to prevent unbounded growth."""
        # Check disk space first
        if self.disk_monitor.is_low_space():
            log("Low disk space detected, running emergency cleanup")
            self.disk_monitor.emergency_cleanup()
            return
        
        prd = self.get_prd()
        stories = prd.get("userStories", [])
        original_count = len(stories)
        prd_modified = False
        
        # ALERT: Very large PRD - may cause prompt overflow
        MAX_TASKS_WARNING = 200
        MAX_TASKS_CRITICAL = 500
        
        if original_count > MAX_TASKS_CRITICAL:
            log(f"CRITICAL: PRD has {original_count} tasks! Running aggressive cleanup...")
            # Keep only 100 completed + all pending
            completed = [s for s in stories if s.get("passes", False)]
            pending = [s for s in stories if not s.get("passes", False)]
            
            if len(completed) > 100:
                # Archive aggressively
                archive_file = self.ralph_dir / "completed_archive.json"
                to_archive = completed[:-100]
                for task in to_archive:
                    task["archivedAt"] = datetime.now().isoformat()
                try:
                    existing = json.loads(archive_file.read_text()) if archive_file.exists() else []
                except Exception:
                    existing = []
                existing.extend(to_archive)
                safe_write_json(archive_file, existing[-1000:])  # Keep max 1000 archived
                
                completed = completed[-100:]
                prd["userStories"] = completed + pending
                prd_modified = True
                log(f"Emergency archive: {len(to_archive)} tasks")
        elif original_count > MAX_TASKS_WARNING:
            log(f"WARNING: PRD has {original_count} tasks - consider cleanup")
        
        # 1. Archive old completed tasks (keep only recent 50)
        completed = [s for s in stories if s.get("passes", False)]
        pending = [s for s in stories if not s.get("passes", False)]
        
        if len(completed) > 50:
            # Archive old completed to separate file
            archive_file = self.ralph_dir / "completed_archive.json"
            existing_archive = []
            if archive_file.exists():
                try:
                    existing_archive = json.loads(archive_file.read_text())
                except Exception:
                    existing_archive = []
            
            # Move oldest completed to archive (keep 50 most recent)
            to_archive = completed[:-50]
            recent_completed = completed[-50:]
            
            # Add timestamp to archived items
            for task in to_archive:
                task["archivedAt"] = datetime.now().isoformat()
            
            existing_archive.extend(to_archive)
            
            # Keep archive manageable (max 500 items)
            if len(existing_archive) > 500:
                existing_archive = existing_archive[-500:]
            
            safe_write_json(archive_file, existing_archive)
            
            # Update PRD with only recent completed + pending
            prd["userStories"] = recent_completed + pending
            prd_modified = True
            log(f"Archived {len(to_archive)} completed tasks")
        
        # 2. Clean completed FIX-* tasks (always remove these)
        fix_count = len([s for s in prd["userStories"] if s.get("id", "").startswith("FIX-") and s.get("passes")])
        prd["userStories"] = [
            s for s in prd.get("userStories", [])
            if not (s.get("id", "").startswith("FIX-") and s.get("passes", False))
        ]
        if fix_count > 0:
            prd_modified = True
            log(f"Cleaned {fix_count} completed FIX tasks")
        
        if prd_modified:
            self.save_prd(prd)
            log(f"PRD: {original_count} -> {len(prd['userStories'])} tasks")
        
        # 2. Rotate progress.txt (runtime log)
        if self.runtime_log.exists():
            lines = self.runtime_log.read_text().split("\n")
            if len(lines) > MAX_PROGRESS_LINES * 1.5:
                header = lines[:10]
                recent = lines[-MAX_PROGRESS_LINES:]
                safe_write_text(self.runtime_log, "\n".join(header + ["", "...(older entries removed)...", ""] + recent))
                log(f"Rotated progress.txt: {len(lines)} -> {len(header) + len(recent) + 3} lines")
        
        # 3. Rotate ralph.log
        log_file = self.ralph_dir / "ralph.log"
        if log_file.exists():
            size = log_file.stat().st_size
            if size > MAX_LOG_SIZE_BYTES:
                content = log_file.read_text()
                safe_write_text(log_file, "...(older logs removed)...\n\n" + content[-200000:])
                log(f"Rotated ralph.log: {size//1024}KB -> ~200KB")
        
        # 4. Rotate model_output.log
        model_log = self.ralph_dir / "model_output.log"
        if model_log.exists():
            size = model_log.stat().st_size
            if size > MAX_LOG_SIZE_BYTES * 2:
                content = model_log.read_text()
                safe_write_text(model_log, "...(older output removed)...\n\n" + content[-500000:])
                log(f"Rotated model_output.log: {size//1024}KB -> ~500KB")
        
        # 5. Archive old iteration files (don't delete - knowledge is valuable!)
        # We compress old iterations to save space while preserving history
        iterations_dir = self.ralph_dir / "iterations"
        archive_dir = self.ralph_dir / "iterations_archive"
        if iterations_dir.exists():
            files = sorted(iterations_dir.glob("iteration_*.json"))
            if len(files) > MAX_ITERATION_FILES:
                # Archive old iterations instead of deleting
                Docker.mkdir(self.container_name, f"{self.CONTAINER_RALPH_DIR}/iterations_archive")
                to_archive = files[:-MAX_ITERATION_FILES + 100]
                
                # Group by date for archiving
                import gzip
                archived_count = 0
                for f in to_archive:
                    try:
                        # Compress and move to archive
                        archive_file = archive_dir / f"{f.name}.gz"
                        with open(f, 'rb') as f_in:
                            with gzip.open(archive_file, 'wb') as f_out:
                                f_out.write(f_in.read())
                        f.unlink()  # Remove original after successful compression
                        archived_count += 1
                    except Exception as e:
                        log(f"Failed to archive {f.name}: {e}")
                
                if archived_count > 0:
                    log(f"Archived {archived_count} old iteration files to iterations_archive/")
        
        # 6. Consolidate knowledge
        self.knowledge.consolidate()
        
        # 7. Clean old backups
        backup_dir = self.ralph_dir / "backups"
        if backup_dir.exists():
            for pattern in ["state_*.json", "prd_*.json"]:
                backups = sorted(backup_dir.glob(pattern))
                for old_backup in backups[:-10]:
                    old_backup.unlink()
    
    def get_progress(self) -> Tuple[int, int]:
        """Get (done_tasks, total_tasks). Blocked tasks count as 'done' for progress."""
        prd = self.get_prd()
        stories = prd.get("userStories", [])
        total = len(stories)
        # Count both passed and blocked as "done" for progress tracking
        done = sum(1 for s in stories if s.get("passes", False) or s.get("blocked", False))
        return (done, total)
    
    def get_iteration_type(self, iteration: int) -> str:
        """
        Determine iteration type with clear priority:
        
        1. task_verify - independent verification of a claimed task (HIGHEST PRIORITY)
        2. add_planning - explicit request to add new feature
        3. planning - no real tasks yet (initial planning)
        4. verification - all tasks done, need to verify project
        5. architect - periodic review (every N iterations)
        6. worker - default, execute tasks
        
        CRITICAL: Once real tasks exist, NEVER go back to planning!
        Architect handles adding/modifying tasks after initial planning.
        """
        config = self.get_config()
        prd = self.get_prd()
        phase = prd.get("phase", "planning")
        stories = prd.get("userStories", [])
        
        # 1. HIGHEST PRIORITY: Task verification pending
        # If a task was claimed done, we need to verify it BEFORE anything else
        pending_verify = config.get("pendingTaskVerification")
        if pending_verify:
            return "task_verify"
        
        # 2. Explicit add_planning request (for new features)
        if phase == "add_planning":
            return "add_planning"
        
        # 3. Initial planning - ONLY if no real tasks exist yet
        #    (PLAN task doesn't count as real task)
        real_tasks = [s for s in stories if s.get("id", "").startswith("TASK-")]
        if not real_tasks:
            # No tasks yet OR phase explicitly set to planning
            return "planning"
        
        # === FROM HERE: Real tasks exist, NEVER return "planning" ===
        # If phase is still "planning", fix it
        if phase == "planning":
            prd["phase"] = "execution"
            self.save_prd(prd)
        
        # 4. Verification - all tasks done (passed OR blocked)
        # A task is "actionable" if it's not passed AND not blocked
        actionable = sum(1 for s in stories if not s.get("passes", False) and not s.get("blocked", False))
        verified = prd.get("verified", False)
        
        if actionable == 0 and not verified:
            return "verification"
        
        # 5. Architect review - periodic (every N iterations)
        architect_interval = config.get("architectInterval", 10)
        if iteration > 0 and iteration % architect_interval == 0:
            return "architect"
        
        # 6. Default - worker executes tasks
        return "worker"
    
    def should_run_condense(self, iteration: int) -> Tuple[bool, str]:
        """Check if context condensation should run before this iteration."""
        config = self.get_config()
        
        # Check if next iteration will be architect
        architect_interval = config.get("architectInterval", 10)
        next_is_architect = iteration > 0 and iteration % architect_interval == 0
        
        # Estimate current context size
        learnings_content = self._read_file("LEARNINGS.md") or ""
        learnings_size = len(learnings_content)
        condensed_size = len(self.condenser.get_condensed_context())
        memory_size = sum(
            len(f.read_text()) for f in (self.ralph_dir / "memory").glob("*.json")
            if f.exists()
        ) if (self.ralph_dir / "memory").exists() else 0
        
        total_context = learnings_size + condensed_size + memory_size
        
        return self.condenser.should_condense(
            iteration, config, total_context, next_is_architect
        )
    
    def build_condense_prompt(self, iteration: int) -> str:
        """Build prompt for context condensation."""
        # Get mission via Docker
        mission = self._read_file("MISSION.md") or ""
        
        # Get recent progress from iteration logs
        iterations_dir = self.ralph_dir / "iterations"
        recent_progress = self.condenser.get_recent_iterations_summary(iterations_dir, 15)
        
        # Get learnings
        learnings = self.get_learnings_summary()
        
        # Get project state
        prd = self.get_prd()
        done, total = self.get_progress()
        project_state = f"""
Tasks: {done}/{total} completed
Phase: {prd.get('phase', 'unknown')}
Verified: {prd.get('verified', False)}

Current tasks:
"""
        for story in prd.get("userStories", [])[:20]:
            status = "✓" if story.get("passes") else "○"
            project_state += f"  {status} [{story.get('id')}] {story.get('title', '')[:50]}\n"
        
        return self.condenser.build_condense_prompt(
            iteration, mission, recent_progress, learnings, project_state
        )
    
    def process_condense_output(self, iteration: int, output: str) -> bool:
        """Process condensation output with verification."""
        condensed = self.condenser.parse_condensed_output(output)
        
        if not condensed:
            log_error(f"Failed to parse condensed output from iteration {iteration}")
            return False
        
        # Get original content for verification
        original_content = ""
        iterations_dir = self.ralph_dir / "iterations"
        if iterations_dir.exists():
            original_content = self.condenser.get_recent_iterations_summary(iterations_dir)
        original_content += "\n" + self.learnings_mgr.get_all()
        
        # Verify condensation preserves critical facts
        is_valid, missing_facts = self.condenser.verify_condensation(
            original_content, condensed, self.semantic, self.metrics
        )
        
        if not is_valid and missing_facts:
            # Add missing facts to condensed
            condensed = self.condenser.enhance_condensed_with_missing(condensed, missing_facts)
            log(f"Condense verification: added {len(missing_facts)} missing facts")
        
        self.condenser.save_condensed(iteration, condensed)
        self.add_progress_entry(iteration, f"Context condensed: {len(condensed)} chars")
        return True
    
    def save_learning(self, category: str, content: str, is_critical: bool = False):
        """Save a learning with semantic deduplication."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if is_critical:
            # Save to critical knowledge
            self.knowledge.add_critical(category, content)
        
        # Use LearningsManager with deduplication
        entry = f"### [{timestamp}] {category}\n{content}"
        added = self.learnings_mgr.add(entry, self.metrics)
        
        if added:
            log(f"Learning saved: {category}")
        else:
            log(f"Learning skipped (duplicate): {category}")
    
    def get_learnings_summary(self) -> str:
        """Get optimized learnings summary."""
        return self.learnings_mgr.get_all()
    
    def get_current_task(self) -> Optional[dict]:
        """Get next task to work on (skips blocked tasks)."""
        prd = self.get_prd()
        completed_ids = {s["id"] for s in prd.get("userStories", []) if s.get("passes", False)}
        blocked_ids = {s["id"] for s in prd.get("userStories", []) if s.get("blocked", False)}
        
        for story in sorted(prd.get("userStories", []), key=lambda x: x.get("priority", 999)):
            task_id = story.get("id", "")
            
            # Skip completed tasks
            if story.get("passes", False):
                continue
            
            # Skip blocked tasks (failed too many times)
            if story.get("blocked", False):
                continue
            
            # Check dependencies (completed OR blocked count as "done" for dependency purposes)
            deps = story.get("dependsOn", [])
            deps_satisfied = all(d in completed_ids or d in blocked_ids for d in deps)
            
            if deps_satisfied:
                return story
        
        return None
    
    def _build_project_summary(self) -> str:
        """Build a summary of current project state."""
        lines = ["## Current Project State\n"]
        
        try:
            result = subprocess.run(
                ["find", str(self.project.workspace), "-type", "f", 
                 "-name", "*.py", "-o", "-name", "*.ts", "-o", "-name", "*.js",
                 "-o", "-name", "*.go", "-o", "-name", "*.rs"],
                capture_output=True, text=True, timeout=5
            )
            files = [f.replace(str(self.project.workspace) + "/", "") 
                    for f in result.stdout.strip().split("\n") if f and ".ralph" not in f][:50]
            if files:
                lines.append(f"**Source files ({len(files)}):**")
                lines.append("```")
                lines.extend(files[:30])
                if len(files) > 30:
                    lines.append(f"... and {len(files)-30} more")
                lines.append("```\n")
        except Exception:
            pass
        
        try:
            result = subprocess.run(
                ["git", "-C", str(self.project.workspace), "log", "--oneline", "-20"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                lines.append("**Recent commits:**")
                lines.append("```")
                lines.append(result.stdout.strip())
                lines.append("```\n")
        except Exception:
            pass
        
        return "\n".join(lines) if len(lines) > 1 else ""
    
    def _get_completed_tasks_summary(self) -> str:
        """Get summary of completed tasks."""
        prd = self.get_prd()
        completed = [s for s in prd.get("userStories", []) if s.get("passes", False)]
        
        if not completed:
            return "(No tasks completed yet)"
        
        lines = [f"**Completed ({len(completed)} tasks):**"]
        for task in completed[-30:]:
            lines.append(f"- [{task.get('id')}] {task.get('title', 'Untitled')}")
        
        if len(completed) > 30:
            lines.insert(1, f"  (showing last 30 of {len(completed)})")
        
        return "\n".join(lines)
    
    def _read_iteration_log(self, log_path: Path) -> str:
        """Read iteration log, supporting both .log and .log.gz files."""
        import gzip
        gz_path = log_path.with_suffix(".log.gz")
        
        if log_path.exists():
            return log_path.read_text()
        elif gz_path.exists():
            with gzip.open(gz_path, 'rt', errors='ignore') as f:
                return f.read()
        return ""
    
    def _get_last_iteration_summary(self, current_iteration: int) -> str:
        """Get brief summary of what previous worker did."""
        if current_iteration <= 1:
            return "(First worker iteration)"
        
        iterations_dir = self.ralph_dir / "iterations"
        prev_iteration = current_iteration - 1
        log_file = iterations_dir / f"iteration_{prev_iteration:04d}.log"
        json_file = iterations_dir / f"iteration_{prev_iteration:04d}.json"
        
        content = self._read_iteration_log(log_file)
        if not content:
            return "(No previous iteration log)"
        
        try:
            content = content[-8000:]  # Last 8KB
            summary_parts = []
            
            # Check if previous iteration was interrupted/timed out
            was_timeout = False
            timeout_reason = ""
            if json_file.exists():
                try:
                    meta = json.loads(json_file.read_text())
                    # Check termination field first (new format)
                    termination = meta.get("termination", "")
                    if termination.startswith("timeout"):
                        was_timeout = True
                        timeout_reason = termination
                    # Fallback to result field (old format)
                    elif "timeout" in meta.get("result", "").lower():
                        was_timeout = True
                        timeout_reason = meta.get("result", "")
                except Exception:
                    pass
            
            # Also check log content for timeout
            if not was_timeout and 'Timeout:' in content:
                was_timeout = True
                timeout_match = re.search(r'Timeout:\s*([^\n]+)', content)
                if timeout_match:
                    timeout_reason = timeout_match.group(1)
            
            if was_timeout:
                summary_parts.append(f"**⚠️ PREVIOUS ITERATION WAS INTERRUPTED** ({timeout_reason})")
                summary_parts.append("The work may be incomplete. Check git status and continue if needed.")
            
            # What task was worked on
            task_match = re.search(r'YOUR TASK:\s*(\S+)', content)
            if task_match:
                summary_parts.append(f"**Task:** {task_match.group(1)}")
            
            # Commands executed
            commands = []
            for pattern in [r'\$ ((?:apt|pip|npm|cargo|make|curl|wget|rustup)[^\n]{0,50})']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                commands.extend(matches[:3])
            if commands:
                summary_parts.append("**Ran:** " + ", ".join(f"`{c[:30]}`" for c in commands[:3]))
            
            # Files modified
            files = re.findall(r'(?:creat|modif|edit|writ)\w*\s+[`"]?([^\s`"]+\.\w{1,4})', content, re.I)
            if files:
                summary_parts.append("**Files:** " + ", ".join(f"`{f}`" for f in list(dict.fromkeys(files))[:4]))
            
            # Errors
            errors = re.findall(r'(?:error|fail|not found)[:\s]+([^\n]{0,60})', content, re.I)
            if errors:
                summary_parts.append(f"**Issue:** {errors[0][:50]}")
            
            # Result
            if 'TASK_DONE:' in content:
                summary_parts.append("**Result:** ✓ Done")
            elif 'STUCK' in content:
                summary_parts.append("**Result:** ✗ Stuck")
            elif was_timeout:
                summary_parts.append("**Result:** ⏱️ Interrupted (continue work)")
            
            return "\n".join(summary_parts) if summary_parts else "(No significant actions)"
        except Exception:
            return "(Could not read previous iteration)"
    
    def build_prompt(self, iteration: int) -> str:
        """Build optimized prompt for iteration.
        
        Template priority:
        1. Project-specific: workspace/.ralph/prompts/{iter_type}.md
        2. Global: templates/ralph/{iter_type}.md
        3. Fallback: templates/ralph/default.md
        """
        iter_type = self.get_iteration_type(iteration)
        
        # 1. Check for project-specific override first
        project_prompt_dir = self.workspace_ralph / "prompts"
        project_prompt_file = project_prompt_dir / f"{iter_type}.md"
        if project_prompt_file.exists():
            template = project_prompt_file.read_text()
            return self._substitute_prompt_vars(template, iteration, iter_type)
        
        # 2. Load from global template directory
        prompt_file = RALPH_PROMPTS_DIR / f"{iter_type}.md"
        if prompt_file.exists():
            template = prompt_file.read_text()
            return self._substitute_prompt_vars(template, iteration, iter_type)
        
        # 3. Fallback: try default.md
        default_file = RALPH_PROMPTS_DIR / "default.md"
        if default_file.exists():
            template = default_file.read_text()
            return self._substitute_prompt_vars(template, iteration, iter_type)
        
        # No template found
        log_error(f"No prompt template found: {prompt_file}")
        return f"Ralph iteration {iteration} ({iter_type}). Check .ralph/ for context."
    
    def _substitute_prompt_vars(self, template: str, iteration: int, iter_type: str) -> str:
        """Substitute variables in prompt template with context optimization."""
        config = self.get_config()
        done, total = self.get_progress()
        task = self.get_current_task()
        mission = self._read_file("MISSION.md") or ""
        task_desc = task.get("description", "") if task else ""
        task_title = task.get("title", "") if task else ""
        
        # Get condensed context if available (from periodic summarization)
        condensed_context = self.condenser.get_condensed_context()
        
        # Semantic context selection: use task-relevant learnings
        # If we have condensed context, prioritize it over raw learnings
        if condensed_context:
            learnings = condensed_context
            # Add recent learnings not yet in condensed context
            recent_learnings = self.learnings_mgr.get_relevant(
                task_title + ' ' + task_desc if task else "",
                max_chars=10000
            )
            if recent_learnings:
                learnings += "\n\n## Recent Learnings\n" + recent_learnings
        elif task and (task_title or task_desc):
            # Use semantic search for relevant learnings
            learnings = self.learnings_mgr.get_relevant(
                task_title + ' ' + task_desc,
                max_chars=RECENT_LEARNINGS_LIMIT
            )
        else:
            learnings = self.learnings_mgr.get_all()[:RECENT_LEARNINGS_LIMIT]
        
        # Get hierarchical memory context (semantic search for relevance)
        query = task_title + ' ' + task_desc if task else ""
        memory_context = self.memory.get_context(max_chars=MEMORY_CONTEXT_LIMIT)
        
        def read_limited(filepath: Path, max_chars: int) -> str:
            """Read file with character limit."""
            if not filepath.exists():
                return ""
            content = filepath.read_text()
            if len(content) > max_chars:
                return f"...(truncated, showing last {max_chars} chars)...\n" + content[-max_chars:]
            return content
        
        # Load sections with size limits
        agents_file = self.ralph_dir / "AGENTS.md"
        agents_section = ""
        if agents_file.exists():
            content = read_limited(agents_file, 5000)
            agents_section = f"## Build & Run Instructions\n\n```\n{content}\n```"
        
        guardrails_file = self.ralph_dir / "guardrails.md"
        guardrails_section = ""
        if guardrails_file.exists():
            content = read_limited(guardrails_file, GUARDRAILS_LIMIT).strip()
            if content and "Add signs" not in content:
                guardrails_section = f"## Guardrails (avoid these mistakes)\n\n{content}"
        
        arch_file = self.ralph_dir / "ARCHITECTURE.md"
        architecture = read_limited(arch_file, ARCHITECTURE_LIMIT) if arch_file.exists() else ""
        
        if len(mission) > MISSION_LIMIT:
            mission = mission[:MISSION_LIMIT] + "\n...(mission truncated)..."
        
        project_summary = self._build_project_summary()
        completed_summary = self._get_completed_tasks_summary()
        
        # For add_planning
        prd = self.get_prd()
        stories = prd.get("userStories", [])
        
        # Collect feature_request tasks as new_feature (they need to be broken down)
        new_feature_parts = []
        existing_tasks_list = []
        
        for s in stories:
            if s.get("type") == "feature_request" and s.get("needs_planning"):
                # This is a pending feature request - add to new_feature
                new_feature_parts.append(f"- {s.get('description', s.get('title', '?'))}")
            else:
                # Regular task - add to existing tasks
                status = "DONE" if s.get("passes") else "PENDING"
                existing_tasks_list.append(f"  [{status}] [{s.get('id', '?')}] {s.get('title', '?')}")
        
        # Also check for explicit newFeature field (legacy)
        if prd.get("newFeature"):
            new_feature_parts.insert(0, prd.get("newFeature"))
        
        new_feature = "\n".join(new_feature_parts) if new_feature_parts else ""
        existing_tasks = "\n".join(existing_tasks_list) if existing_tasks_list else "(No existing tasks)"
        
        # Get epoch context for long-running projects
        epoch_context = self.get_epoch_context()
        
        # Build sections dict for context optimization
        sections = {
            "mission": mission,
            "learnings": learnings,
            "architecture": architecture,
            "guardrails": guardrails_section,
            "agents": agents_section,
            "project_summary": project_summary,
            "completed_tasks": completed_summary,
            "epochs": epoch_context,  # Historical milestones
            "memory": memory_context,  # Hierarchical memory (hot/warm/cold)
        }
        
        # Dynamic priority order based on iteration type
        # Higher = more important, will be included first
        if iter_type == "planning":
            # Planning needs mission focus, less learnings
            priorities = {
                "mission": 100,
                "project_summary": 90,
                "epochs": 75,
                "architecture": 70,
                "guardrails": 60,
                "memory": 55,           # Recent iterations context
                "learnings": 50,
                "agents": 40,
                "completed_tasks": 30,
            }
        elif iter_type == "worker":
            # Worker needs task context and learnings to avoid mistakes
            priorities = {
                "guardrails": 100,
                "learnings": 95,
                "memory": 92,           # Very important - recent work context
                "mission": 80,
                "agents": 75,
                "architecture": 60,
                "completed_tasks": 50,
                "project_summary": 40,
                "epochs": 30,
            }
        elif iter_type == "architect":
            # Architect needs patterns and architecture focus
            priorities = {
                "architecture": 100,
                "mission": 90,
                "epochs": 88,
                "memory": 87,           # Recent iterations help review
                "learnings": 85,
                "completed_tasks": 80,
                "guardrails": 70,
                "project_summary": 60,
                "agents": 40,
            }
        elif iter_type == "verification":
            # Verification needs mission and completed work
            priorities = {
                "mission": 100,
                "completed_tasks": 95,
                "memory": 90,           # What was recently done
                "epochs": 85,
                "learnings": 80,
                "guardrails": 70,
                "agents": 60,
                "architecture": 50,
                "project_summary": 40,
            }
        elif iter_type == "task_verify":
            # Task verification needs learnings, mission, and task context
            priorities = {
                "learnings": 100,       # Most important - check for contradictions
                "mission": 95,          # What was supposed to be achieved
                "memory": 90,           # Recent context
                "agents": 80,           # How to check things
                "project_summary": 70,  # Current state
                "completed_tasks": 60,
                "architecture": 40,
                "guardrails": 30,
                "epochs": 20,
            }
        else:
            # Default priorities
            priorities = {
                "mission": 100,
                "guardrails": 90,
                "learnings": 80,
                "memory": 75,
                "architecture": 70,
                "epochs": 65,
                "agents": 60,
                "completed_tasks": 50,
                "project_summary": 40,
            }
        
        # Use context manager to optimize
        optimized_context = self.context_manager.optimize_prompt(sections, priorities)
        
        # Get previous attempts for current task
        task_id = task.get("id", "") if task else ""
        previous_attempts = self.stuck_detector.get_task_attempts(task_id) if task_id else ""
        
        # Get last iteration summary (what previous worker did)
        last_iteration_summary = self._get_last_iteration_summary(iteration)
        
        # For FIX tasks: get history of similar error fixes (file-based + git-based)
        fix_history = ""
        if task_id.startswith("FIX"):
            fix_history = self.stuck_detector.get_fix_history(task_desc)
            
            # Also get git-based context for this FIX task
            git_fix_context = self.git_handoff.get_fix_context(task_id)
            if git_fix_context:
                git_history = "\n[Git history for this fix]:\n"
                for ctx in git_fix_context[:5]:  # Limit to 5 most recent
                    if ctx.get("commit"):
                        git_history += f"- Commit {ctx['commit']}: {ctx.get('message', '')}\n"
                    elif ctx.get("type") == "context":
                        data = ctx.get("data", {})
                        if data.get("attempts"):
                            git_history += f"- Previous attempts: {len(data['attempts'])}\n"
                fix_history += git_history
        
        # QMD semantic search info
        qmd_mgr = QMDManager(self.project)
        qmd_info = ""
        if qmd_mgr.is_enabled:
            qmd_setup_marker = self.project.workspace / ".qmd_installed"
            if hasattr(self, 'qmd_needs_setup') and self.qmd_needs_setup:
                qmd_info = """
## QMD Semantic Search (FIRST RUN - SETUP REQUIRED)
Run this command FIRST to enable semantic code search:
```bash
bash /workspace/.qmd_setup.sh
```
After setup, mark it complete:
```bash
touch /workspace/.qmd_installed
```
"""
            elif qmd_setup_marker.exists():
                qmd_info = """
## Semantic Code Search Available (QMD)
You have access to QMD for semantic code search. Use when grep/find isn't enough:
- `qmd search "keyword"` - fast BM25 keyword search
- `qmd vsearch "how to handle auth"` - semantic vector search
- `qmd query "database connection pattern"` - hybrid search with reranking (best quality)
- `qmd get "path/to/file.py"` - get document by path
"""
        
        # For task_verify: get pending verification info
        pending_verify = config.get("pendingTaskVerification", {})
        verify_task_id = pending_verify.get("taskId", "")
        verify_task_title = pending_verify.get("taskTitle", "")
        verify_task_desc = pending_verify.get("taskDescription", "")
        
        replacements = {
            "${iteration}": str(iteration),
            "${iter_type}": iter_type,
            "${done_tasks}": str(done),
            "${total_tasks}": str(total),
            "${done}": str(done),
            "${total}": str(total),
            "${task_id}": verify_task_id if iter_type == "task_verify" and verify_task_id else (task.get("id", "???") if task else "???"),
            "${task_title}": verify_task_title if iter_type == "task_verify" and verify_task_title else (task.get("title", "No task") if task else "No task"),
            "${task_description}": verify_task_desc if iter_type == "task_verify" and verify_task_desc else task_desc,
            "${mission_content}": mission,
            "${mission}": mission,
            "${learnings}": learnings,
            "${project_name}": self.project.name,
            "${agents_section}": agents_section,
            "${guardrails_section}": guardrails_section,
            "${architecture}": architecture,
            "${project_summary}": project_summary,
            "${completed_tasks}": completed_summary,
            "${new_feature}": new_feature,
            "${existing_tasks}": existing_tasks if existing_tasks else "(no existing tasks)",
            "${epoch_context}": epoch_context if epoch_context else "",
            "${memory_context}": memory_context if memory_context else "",
            "${previous_attempts}": previous_attempts,
            "${last_iteration}": last_iteration_summary,
            "${fix_history}": fix_history,
            "${qmd_info}": qmd_info,
        }
        
        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)
        
        # Validate - check for unresolved variables
        unresolved = re.findall(r'\$\{[a-zA-Z_]+\}', result)
        if unresolved:
            log_error(f"Unresolved variables in prompt: {unresolved}")
        
        # Log context usage
        self.context_manager.log_usage(result)
        
        return result
    
    def parse_output(self, output: str) -> str:
        """
        Parse Ralph output and return result type.
        
        STRICT: Only accept <ralph>...</ralph> tagged outputs.
        No fallback parsing to prevent false positives.
        """
        output = strip_ansi(output)
        
        # STRICT: Only look for <ralph>...</ralph> tags
        ralph_match = re.search(r'<ralph>\s*([^<]+?)\s*</ralph>', output, re.IGNORECASE | re.DOTALL)
        if ralph_match:
            tag_content = ralph_match.group(1).strip()
            
            # Parse tag content
            if re.match(r'^VERIFIED$', tag_content, re.IGNORECASE):
                return "VERIFIED"
            
            # TASK_VERIFIED:TASK-XXX - independent verification passed
            if re.match(r'^TASK[_\s]*VERIFIED', tag_content, re.IGNORECASE):
                task_match = re.search(r'TASK[_\s]*VERIFIED[:\s]*([A-Za-z0-9_.-]+)', tag_content, re.IGNORECASE)
                if task_match:
                    return f"TASK_VERIFIED:{task_match.group(1)}"
            
            # TASK_REJECTED:TASK-XXX:reason - independent verification failed
            if re.match(r'^TASK[_\s]*REJECTED', tag_content, re.IGNORECASE):
                # Format: TASK_REJECTED:TASK-XXX:reason or TASK_REJECTED:TASK-XXX
                reject_match = re.search(r'TASK[_\s]*REJECTED[:\s]*([A-Za-z0-9_.-]+)(?:[:\s]+(.+))?', tag_content, re.IGNORECASE)
                if reject_match:
                    task_id = reject_match.group(1)
                    reason = reject_match.group(2) or "No reason given"
                    return f"TASK_REJECTED:{task_id}:{reason[:300]}"
            
            if re.match(r'^NEEDS_WORK', tag_content, re.IGNORECASE):
                reason = re.sub(r'^NEEDS_WORK[:\s]*', '', tag_content, flags=re.IGNORECASE)
                return f"NEEDS_WORK:{reason[:200] or 'verification failed'}"
            
            if re.match(r'^ARCHITECT_DONE$', tag_content, re.IGNORECASE):
                return "ARCHITECT_DONE"
            
            if re.match(r'^TASK[_\s]*DONE', tag_content, re.IGNORECASE):
                task_match = re.search(r'TASK[_\s]*DONE[:\s]*([A-Za-z0-9_.-]+)', tag_content, re.IGNORECASE)
                if task_match:
                    return f"TASK_DONE:{task_match.group(1)}"
            
            if re.match(r'^STUCK$', tag_content, re.IGNORECASE):
                return "STUCK"
            
            if re.match(r'^ALL[_\s]?COMPLETE$', tag_content, re.IGNORECASE):
                return "ALL_COMPLETE"
        
        # NO FALLBACK - strict parsing only
        return "UNKNOWN"
    
    def _check_conversation_finished(self, container_name: str) -> Optional[str]:
        """Check if OpenHands conversation has finished inside container."""
        try:
            result = subprocess.run(
                ["docker", "exec", container_name, "bash", "-c",
                 "ls -t /root/.openhands/conversations/*/metadata.json 2>/dev/null | head -1"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                return None
            
            conv_dir = result.stdout.strip().replace("/metadata.json", "")
            
            result = subprocess.run(
                ["docker", "exec", container_name, "bash", "-c",
                 f"cat {conv_dir}/metadata.json 2>/dev/null | grep -o '\"execution_status\":\"[^\"]*\"'"],
                capture_output=True, text=True, timeout=10
            )
            
            if "finished" in result.stdout:
                result = subprocess.run(
                    ["docker", "exec", container_name, "bash", "-c",
                     f"cat {conv_dir}/*.json 2>/dev/null | grep -o '<ralph>[^<]*</ralph>' | tail -1"],
                    capture_output=True, text=True, timeout=10
                )
                if result.stdout.strip():
                    return result.stdout.strip()
                return "<ralph>UNKNOWN</ralph>"
            
            return None
        except Exception:
            return None
    
    def request_pause(self):
        """Request Ralph to pause."""
        self.update_config("pauseRequested", True)
        self._stop_event.set()
        # Save state for resume
        self.state_manager.save_state(self)
    
    def request_stop(self):
        """Request Ralph to stop."""
        self.update_config("status", "stopped")
        self.update_config("pauseRequested", True)
        self._stop_event.set()
        # Save final state
        self.state_manager.save_state(self)
    
    def is_stop_requested(self) -> bool:
        """Check if stop was requested (thread-safe)."""
        return self._stop_event.is_set()
    
    def _extract_conversation(self, container_name: str) -> str:
        """Extract the last meaningful AI response from OpenHands conversation."""
        try:
            result = subprocess.run(
                ["docker", "exec", container_name, "bash", "-c",
                 "ls -t /root/.openhands/conversations/*/events.json 2>/dev/null | head -1"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                return ""
            
            events_file = result.stdout.strip()
            
            result = subprocess.run(
                ["docker", "exec", container_name, "cat", events_file],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                return ""
            
            agent_messages = []
            
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    
                    if event.get("message") and event.get("source") == "agent":
                        msg = event.get("message", "")
                        if msg:
                            agent_messages.append(msg)
                    
                    args = event.get("args", {})
                    if args.get("content") and args.get("role") == "assistant":
                        content = args.get("content", "")
                        if content:
                            agent_messages.append(content)
                    
                    if args.get("thought"):
                        thought = args.get("thought", "")
                        if thought and len(thought) > 30:
                            agent_messages.append(thought)
                    
                    if event.get("action") == "finish":
                        outputs = args.get("outputs", {})
                        if outputs.get("content"):
                            agent_messages.append(outputs.get("content"))
                    
                    if event.get("observation") and event.get("source") == "agent":
                        obs_content = event.get("content", "")
                        if obs_content and len(obs_content) > 50:
                            agent_messages.append(obs_content)
                            
                except json.JSONDecodeError:
                    continue
            
            if agent_messages:
                last_msg = agent_messages[-1]
                last_msg = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', last_msg)
                last_msg = re.sub(r'\[[\d;]*[mHJK]', '', last_msg)
                return last_msg[:8000]
            
            all_events = result.stdout.strip().split("\n")
            for line in reversed(all_events[-20:]):
                try:
                    event = json.loads(line)
                    for key in ['message', 'content', 'thought', 'text']:
                        val = event.get(key) or event.get("args", {}).get(key)
                        if val and isinstance(val, str) and len(val) > 100:
                            clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', val)
                            return clean[:8000]
                except Exception:
                    continue
            
            return ""
            
        except Exception as e:
            return ""
    
    def _filter_terminal_output(self, output: str) -> str:
        """Filter terminal noise from stdout."""
        if not output:
            return ""
        
        output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', output)
        output = re.sub(r'\[[\d;]*[mHJK]', '', output)
        
        lines = output.split('\n')
        filtered = []
        
        skip_patterns = [
            r'^Working \(\d+s',
            r'^\| Type your message',
            r'^\[Ctrl\\+',
            r'^\d+K \• \$',
            r'^::? Working',
            r'ESC: pause',
            r'cache \d+%',
            r'^\s*$',
            r'^Type your message',
            r'@mention a file',
            r'for commands$',
        ]
        
        for line in lines:
            line = line.rstrip()
            skip = False
            for pattern in skip_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            if not skip and line:
                filtered.append(line)
        
        return '\n'.join(filtered[-100:])
    
    def add_progress_entry(self, iteration: int, message: str):
        """Add entry to progress files via Docker."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Write to git-tracked PROGRESS.md (condensed, important events)
        important_keywords = ['TASK_DONE', 'VERIFIED', 'STUCK', 'FIX', 'Starting', 'Complete', 'Error']
        is_important = any(kw in message for kw in important_keywords)
        
        if is_important:
            self._append_file("PROGRESS.md", f"- [{timestamp}] Iter {iteration}: {message}\n")
        
        # Write to runtime log (detailed, all entries)
        self._append_file("progress.txt", f"[{timestamp}] [Iteration {iteration}] {message}\n")
    
    def get_context_stats(self) -> dict:
        """Get context window usage statistics."""
        return self.context_manager.get_stats()
    
    def update_status(self, iteration: int = 0):
        """Update git-tracked STATUS.md with current project state."""
        try:
            prd = self.get_prd()
            stories = prd.get("userStories", [])
            done = sum(1 for s in stories if s.get("passes"))
            total = len(stories)
            
            config = self.get_config()
            status = config.get("status", "running")
            
            # Calculate completion percentage
            completion = (done / total * 100) if total > 0 else 0
            
            # Generate status content
            content = f"""# Project Status

**Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Iteration:** {iteration}
**Status:** {status}

## Progress

- Tasks completed: {done}/{total}
- Completion: {completion:.1f}%

## Recent Tasks

"""
            # Add recent task status
            for story in stories[-10:]:  # Last 10 tasks
                status_icon = "✅" if story.get("passes") else "⏳"
                content += f"- {status_icon} {story.get('id', '?')}: {story.get('title', 'Untitled')}\n"
            
            self._write_file("STATUS.md", content)
        except Exception:
            pass  # Status update is best-effort
    
    def commit_workspace_ralph(self, message: str = "Update project state"):
        """Commit changes in workspace/.ralph/ to git."""
        try:
            workspace = self.project.workspace
            
            # Stage .ralph directory
            subprocess.run(
                ['git', 'add', '.ralph/'],
                capture_output=True, cwd=workspace, timeout=10
            )
            
            # Check if there are changes to commit
            result = subprocess.run(
                ['git', 'diff', '--cached', '--quiet'],
                capture_output=True, cwd=workspace, timeout=5
            )
            
            if result.returncode != 0:  # There are staged changes
                subprocess.run(
                    ['git', 'commit', '-m', f'[Ralph:State] {message}'],
                    capture_output=True, cwd=workspace, timeout=30
                )
                return True
            return False
        except Exception:
            return False
    
    # =========================================================================
    # EPOCH MANAGEMENT - Track major project milestones
    # =========================================================================
    
    def check_epoch_milestone(self, iteration: int) -> bool:
        """
        Check if we've reached a major milestone that should trigger epoch save.
        
        Milestones:
        - First verification completed
        - Every 100 iterations
        - Phase transitions (planning -> execution)
        - Major feature completion (50%+ tasks done)
        """
        config = self.get_config()
        last_epoch = config.get("lastEpochIteration", 0)
        current_epoch = config.get("currentEpoch", 0)
        
        # Don't save epochs too frequently
        if iteration - last_epoch < 25:
            return False
        
        prd = self.get_prd()
        done, total = self.get_progress()
        
        # Check milestones
        milestones = [
            # First verification
            prd.get("verified") and current_epoch == 0,
            # Every 100 iterations
            iteration > 0 and iteration % 100 == 0,
            # 50% completion milestone
            total > 10 and done >= total * 0.5 and config.get("50pctEpochSaved") != True,
            # 75% completion milestone
            total > 10 and done >= total * 0.75 and config.get("75pctEpochSaved") != True,
        ]
        
        return any(milestones)
    
    def save_epoch(self, iteration: int, reason: str = "milestone") -> bool:
        """
        Save epoch snapshot - a full state capture at a major milestone.
        
        Epochs help with:
        - Understanding project history
        - Recovering from bad states
        - Providing context for new features
        """
        config = self.get_config()
        current_epoch = config.get("currentEpoch", 0) + 1
        
        epoch_dir = self.ralph_dir / "epochs"
        # Directory created by RalphManager via Docker
        
        try:
            done, total = self.get_progress()
            prd = self.get_prd()
            
            # Build epoch summary
            completed_tasks = [s for s in prd.get("userStories", []) if s.get("passes")]
            task_titles = [t.get("title", "Untitled") for t in completed_tasks[-20:]]
            
            epoch_data = {
                "epoch": current_epoch,
                "iteration": iteration,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "progress": {"done": done, "total": total},
                "phase": prd.get("phase", "unknown"),
                "verified": prd.get("verified", False),
                "recentTasks": task_titles,
                "contextStats": self.get_context_stats(),
            }
            
            # Save epoch metadata
            epoch_file = epoch_dir / f"epoch_{current_epoch:03d}.json"
            safe_write_json(epoch_file, epoch_data)
            
            # Save epoch summary for context
            summary_file = epoch_dir / f"epoch_{current_epoch:03d}_summary.md"
            summary_content = f"""# Epoch {current_epoch} Summary

## Milestone
- **Iteration**: {iteration}
- **Reason**: {reason}
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- **Progress**: {done}/{total} tasks ({done*100//total if total > 0 else 0}%)

## Recent Completed Tasks
{chr(10).join(f"- {t}" for t in task_titles)}

## Key Learnings
{self._extract_recent_learnings(500)}

## Architecture Notes
{self._extract_architecture_summary(300)}
"""
            safe_write_text(summary_file, summary_content)
            
            # Update config
            self.update_config("currentEpoch", current_epoch)
            self.update_config("lastEpochIteration", iteration)
            
            # Mark milestone flags
            if done >= total * 0.5 and total > 10:
                self.update_config("50pctEpochSaved", True)
            if done >= total * 0.75 and total > 10:
                self.update_config("75pctEpochSaved", True)
            
            log(f"Saved epoch {current_epoch} at iteration {iteration}: {reason}")
            return True
            
        except Exception as e:
            log_error(f"Failed to save epoch: {e}")
            return False
    
    def _extract_recent_learnings(self, max_chars: int) -> str:
        """Extract recent learnings for epoch summary."""
        content = self._read_file("LEARNINGS.md")
        if not content:
            return "(No learnings yet)"
        
        if len(content) > max_chars:
            return content[-max_chars:]
        return content
    
    def _extract_architecture_summary(self, max_chars: int) -> str:
        """Extract architecture notes for epoch summary."""
        content = self._read_file("ARCHITECTURE.md")
        if not content:
            return "(No architecture notes)"
        
        if len(content) > max_chars:
            return content[-max_chars:]
        return content
    
    def get_epoch_context(self) -> str:
        """Get context from recent epochs for prompts."""
        epoch_dir = self.ralph_dir / "epochs"
        if not epoch_dir.exists():
            return ""
        
        summaries = sorted(epoch_dir.glob("epoch_*_summary.md"))[-3:]  # Last 3 epochs
        if not summaries:
            return ""
        
        parts = ["## Previous Milestones\n"]
        for s in summaries:
            try:
                content = s.read_text()
                # Extract just the key info
                lines = content.split('\n')[:15]  # First 15 lines
                parts.append('\n'.join(lines) + "\n---\n")
            except Exception:
                pass
        
        return '\n'.join(parts)
    
    def add_pending_task_while_running(self, task: str) -> bool:
        """Add a task to queue while Ralph is running (for next iteration)."""
        return self.state_manager.add_pending_task(task)
    
    def process_pending_tasks(self) -> int:
        """Process pending tasks - add placeholder tasks that trigger add_planning.
        
        Pending tasks are stored as JSON with description and priority.
        We create a placeholder ADD_FEATURE task that will trigger add_planning phase
        to properly break down the feature into subtasks.
        """
        pending = self.state_manager.get_pending_tasks()
        if not pending:
            return 0
        
        prd = self.get_prd()
        stories = prd.get("userStories", [])
        
        # Calculate priorities
        min_prio = min([s.get("priority", 100) for s in stories] or [100])
        max_prio = max([s.get("priority", 0) for s in stories] or [0])
        
        added = 0
        for i, task_raw in enumerate(pending):
            # Parse task data (new format: JSON with description and priority)
            try:
                task_data = json.loads(task_raw)
                description = task_data.get("description", task_raw)
                priority_level = task_data.get("priority", "normal")
            except (json.JSONDecodeError, TypeError):
                # Old format: plain string
                description = task_raw
                priority_level = "normal"
            
            # Calculate actual priority based on level
            if priority_level == "high":
                # Insert before current tasks
                priority = min_prio - 10 + i
            elif priority_level == "low":
                # Insert after all tasks
                priority = max_prio + 100 + i
            else:  # normal
                # Insert after current tasks
                priority = max_prio + 10 + i
            
            # Create placeholder task - will trigger add_planning to break it down
            task_id = f"ADD_FEATURE-{int(time.time())}-{i}"
            
            # Short title for display, full description for planning
            title_preview = description[:50] + "..." if len(description) > 50 else description
            
            prd["userStories"].append({
                "id": task_id,
                "title": f"[Feature Request] {title_preview}",
                "description": description,
                "passes": False,
                "priority": priority,
                "type": "feature_request",  # Special type that triggers add_planning
                "needs_planning": True  # Flag to break down into subtasks
            })
            added += 1
            log(f"Added pending task: {task_id} (priority: {priority_level} -> {priority})")
        
        if added > 0:
            # Set phase to add_planning so next iteration breaks down the features
            prd["phase"] = "add_planning"
            self.save_prd(prd)
            self.state_manager.clear_pending_tasks()
        
        return added
    
    def _check_conversation_finished_host(self, start_time: float) -> bool:
        """Check if OpenHands conversation has finished by checking metadata.json on host."""
        try:
            # Check conversations directory
            conv_dir = self.project.workspace / ".openhands" / "conversations"
            if not conv_dir.exists():
                return False
            
            # Find the most recent conversation
            conv_files = list(conv_dir.glob("*/metadata.json"))
            if not conv_files:
                return False
            
            # Sort by modification time
            conv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            for metadata_file in conv_files:
                try:
                    mtime = metadata_file.stat().st_mtime
                    # Only check conversations started after our start_time
                    if mtime > start_time:
                        metadata = json.loads(metadata_file.read_text())
                        # Check if conversation is finished
                        if metadata.get("status") == "finished":
                            return True
                except Exception:
                    continue
            
            return False
        except Exception:
            return False



# =============================================================================
# RETRY MANAGER
# =============================================================================

class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.attempt_history: List[dict] = []
    
    def execute(self, operation: callable, *args, **kwargs) -> Tuple[bool, Any]:
        """
        Execute operation with retry logic.
        
        Returns:
            (success, result_or_error)
        """
        for attempt in range(self.max_retries):
            try:
                result = operation(*args, **kwargs)
                self.attempt_history.append({
                    "attempt": attempt + 1,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                })
                return True, result
            except Exception as e:
                self.attempt_history.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), MAX_DELAY)
                    log(f"Retry {attempt + 1}/{self.max_retries} after {delay}s: {e}")
                    time.sleep(delay)
                else:
                    return False, e
        
        return False, "Max retries exceeded"
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        if not self.attempt_history:
            return {"total": 0, "success_rate": 0}
        
        total = len(self.attempt_history)
        successes = sum(1 for a in self.attempt_history if a.get("success"))
        
        return {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": successes / total if total > 0 else 0
        }


# =============================================================================
# TUI WIDGETS
# =============================================================================

class ProjectCard(Static):
    """Widget displaying project info."""
    
    def __init__(self, project: Project, **kwargs):
        super().__init__(**kwargs)
        self.project = project
    
    def compose(self) -> ComposeResult:
        p = self.project
        status = p.get_container_status()
        status_icons = {"running": "[green]*[/]", "stopped": "[dim]*[/]", "none": "[dim]o[/]"}
        status_icon = status_icons.get(status, "?")
        
        bg_indicator = ""
        if status == "running" and Docker.is_background_running(p):
            bg_indicator = "[cyan]BG[/]"
        
        ralph_status = ""
        if p.has_ralph:
            config = p.get_ralph_config()
            r_status = config.get("status", "")
            done, total = p.get_ralph_progress()
            
            # Verify process is actually running if status says "running"
            if r_status == "running":
                # Quick check if daemon is running in container
                actually_running = False
                try:
                    actually_running = Docker.is_ralph_daemon_running(p)
                except Exception:
                    pass
                
                if actually_running:
                    ralph_status = f"[blue]R[{done}/{total}][/]"
                else:
                    ralph_status = "[yellow]STALE[/]"
            elif r_status == "complete":
                ralph_status = "[green]OK[/]"
            elif r_status:
                ralph_status = "[yellow]P[/]"
        
        llm = p.get_llm_model()
        if len(llm) > 20:
            llm = llm[:17] + "..."
        
        indicators = " ".join(filter(None, [bg_indicator, ralph_status]))
        if indicators:
            indicators = f" {indicators}"
        
        yield Static(f"{status_icon} {p.name}{indicators} [dim]({llm})[/]", markup=True)


# =============================================================================
# TUI SCREENS
# =============================================================================

class NewProjectScreen(ModalScreen):
    """Screen for creating new project."""
    
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    
    def compose(self) -> ComposeResult:
        templates = ProjectManager.list_llm_templates() or ["(no templates)"]
        
        with Container(id="dialog"):
            yield Label("+ Create New Project", id="dialog-title")
            yield Label("Project name:")
            yield Input(placeholder="my-project", id="name-input")
            yield Label("LLM Template:")
            yield Select([(t, t) for t in templates], id="template-select", value=templates[0])
            with Horizontal(classes="switch-row"):
                yield Switch(value=True, id="pull-switch")
                yield Static(" Pull latest Docker image (recommended)")
            yield Static("[dim]Pull ensures you have the newest OpenHands version[/]", markup=True)
            yield Static("[dim]QMD semantic search is enabled by default for all projects[/]", markup=True)
            with Horizontal(id="dialog-buttons"):
                yield Button("Create", variant="primary", id="btn-create")
                yield Button("Cancel", id="btn-cancel")
    
    def action_cancel(self):
        self.dismiss(None)
    
    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-create":
            name = self.query_one("#name-input", Input).value.strip()
            template = self.query_one("#template-select", Select).value
            pull = self.query_one("#pull-switch", Switch).value
            if name:
                # QMD is now always enabled by default
                self.dismiss({"name": name, "template": template, "pull": pull, "qmd": True})
            else:
                self.notify("Enter project name", severity="warning")


class RalphConfigScreen(ModalScreen):
    """Screen for configuring Ralph.
    
    Modes:
    - running: Ralph is actively working → View/Pause/Stop/Add Task
    - complete: All tasks done → View Log/New Mission
    - stopped: Interrupted, has pending tasks → View Log/Resume/Add Task/New Mission  
    - new: First run → Enter task/Start
    """
    
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    
    def __init__(self, project: Project, mode: str = "new", **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.mode = mode  # "running", "complete", "stopped", "new"
    
    def compose(self) -> ComposeResult:
        titles = {
            "running": "Ralph Running",
            "complete": "Ralph Complete", 
            "stopped": "Continue Ralph",
            "new": "Start Ralph"
        }
        title = titles.get(self.mode, "Ralph")
        
        with Container(id="dialog"):
            yield Label(f"* {title}: {self.project.name}", id="dialog-title")
            
            if self.mode == "running":
                # ═══════════════════════════════════════════════════════════
                # RUNNING: Ralph is working now
                # ═══════════════════════════════════════════════════════════
                ralph = RalphManager(self.project)
                config = ralph.get_config()
                iteration = config.get("currentIteration", 0)
                done, total = ralph.get_progress()
                
                yield Static(f"[green]● Ralph is running[/]", markup=True)
                yield Static(f"[dim]Iteration: {iteration} | Progress: {done}/{total} tasks[/]", markup=True)
                yield Static("")
                
                with Horizontal(classes="ralph-actions"):
                    yield Button("View Monitor", variant="primary", id="btn-view-monitor")
                    yield Button("Pause", variant="warning", id="btn-pause")
                    yield Button("Stop", variant="error", id="btn-stop")
                
                yield Static("")
                yield Static("[dim]───── Add task to queue ─────[/]", markup=True)
                yield Label("New task/feature (will be split into subtasks):")
                yield TextArea(id="task-input")
                
                yield Label("Priority:")
                yield Select([
                    ("🔴 High - Do next", "high"),
                    ("🟡 Normal - After current tasks", "normal"),
                    ("🟢 Low - When nothing else", "low")
                ], id="priority-select", value="normal")
                
                with Horizontal(id="dialog-buttons"):
                    yield Button("Add Task", variant="success", id="btn-add-task")
                    yield Button("Close", id="btn-cancel")
            
            elif self.mode == "complete":
                # ═══════════════════════════════════════════════════════════
                # COMPLETE: All tasks done
                # ═══════════════════════════════════════════════════════════
                ralph = RalphManager(self.project)
                done, total = ralph.get_progress()
                
                yield Static(f"[green]✓ All tasks completed ({done}/{total})[/]", markup=True)
                yield Static("[dim]Knowledge preserved: LEARNINGS.md, ARCHITECTURE.md[/]", markup=True)
                yield Static("")
                
                with Horizontal(classes="ralph-actions"):
                    yield Button("View Log", id="btn-view-log")
                
                yield Static("")
                yield Static("[dim]───── Start new mission (keeps knowledge) ─────[/]", markup=True)
                yield Label("New mission:")
                yield TextArea(id="task-input", classes="task-input-large")
                
                with Horizontal(id="dialog-buttons"):
                    yield Button("Start New Mission", variant="primary", id="btn-new-mission")
                    yield Button("Close", id="btn-cancel")
            
            elif self.mode == "stopped":
                # ═══════════════════════════════════════════════════════════
                # STOPPED: Has pending tasks, can resume or add more
                # ═══════════════════════════════════════════════════════════
                ralph = RalphManager(self.project)
                done, total = ralph.get_progress()
                pending = total - done
                
                yield Static(f"[yellow]● Stopped ({done}/{total} done, {pending} pending)[/]", markup=True)
                yield Static("[dim]Knowledge preserved[/]", markup=True)
                yield Static("")
                
                with Horizontal(classes="ralph-actions"):
                    yield Button("View Log", id="btn-view-log")
                    yield Button("Resume", variant="success", id="btn-resume")
                
                yield Static("")
                yield Static("[dim]───── Add more work ─────[/]", markup=True)
                yield Label("Additional task/feature (optional):")
                yield TextArea(id="task-input")
                
                yield Label("Priority:")
                yield Select([
                    ("🔴 High - Do next", "high"),
                    ("🟡 Normal - After current tasks", "normal"),
                    ("🟢 Low - When nothing else", "low")
                ], id="priority-select", value="normal")
                
                yield Static("")
                yield Static("[dim]───── OR start fresh ─────[/]", markup=True)
                
                with Horizontal(id="dialog-buttons"):
                    yield Button("Add & Resume", variant="primary", id="btn-add-resume")
                    yield Button("New Mission", variant="warning", id="btn-new-mission")
                    yield Button("Cancel", id="btn-cancel")
            
            else:
                # ═══════════════════════════════════════════════════════════
                # NEW: First run
                # ═══════════════════════════════════════════════════════════
                yield Label("Task/Mission Description:")
                yield TextArea(id="task-input", classes="task-input-large")
                
                yield Static("")
                yield Static("[dim]Advanced settings (optional):[/]", markup=True)
                
                yield Label("Max Iterations (0 = infinite):")
                yield Input(value="0", id="max-iter-input")
                
                yield Label("Architect Review Every N iterations:")
                yield Input(value="10", id="arch-interval-input")
                
                yield Label("Context Summarization Every N iterations:")
                yield Input(value="15", id="condense-interval-input")
                
                with Horizontal(id="dialog-buttons"):
                    yield Button("Start", variant="primary", id="btn-ralph-start")
                    yield Button("Cancel", id="btn-cancel")
    
    def action_cancel(self):
        self.dismiss(None)
    
    def _get_settings(self) -> dict:
        """Get settings from inputs if available."""
        settings = {
            "max_iterations": 0,
            "architect_interval": 10,
            "condense_interval": 15
        }
        try:
            max_iter = self.query_one("#max-iter-input", Input).value.strip()
            settings["max_iterations"] = int(max_iter) if max_iter.isdigit() else 0
        except Exception:
            pass
        try:
            arch = self.query_one("#arch-interval-input", Input).value.strip()
            settings["architect_interval"] = int(arch) if arch.isdigit() else 10
        except Exception:
            pass
        try:
            cond = self.query_one("#condense-interval-input", Input).value.strip()
            settings["condense_interval"] = int(cond) if cond.isdigit() else 15
        except Exception:
            pass
        return settings
    
    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses. MUST stop event to prevent bubbling to main app!"""
        event.stop()  # Critical: prevent event from bubbling to OpenHandsApp
        
        btn = event.button.id
        
        if btn == "btn-cancel":
            self.dismiss(None)
        
        # ═══════════════════════════════════════════════════════════
        # RUNNING MODE
        # ═══════════════════════════════════════════════════════════
        elif btn == "btn-view-monitor":
            self.dismiss({"action": "view_monitor"})
        
        elif btn == "btn-add-task":
            task = self.query_one("#task-input", TextArea).text.strip()
            if not task:
                self.notify("Enter task description", severity="warning")
                return
            priority = self.query_one("#priority-select", Select).value
            self.dismiss({"action": "add_task", "task": task, "priority": priority})
        
        elif btn == "btn-pause":
            ralph = RalphManager(self.project)
            ralph.update_config("status", "paused")
            self.notify("Ralph paused")
            self.dismiss(None)
        
        elif btn == "btn-stop":
            ralph = RalphManager(self.project)
            ralph.update_config("status", "stopped")
            # Stop daemon in container
            try:
                Docker.stop_ralph_daemon(self.project)
            except Exception:
                pass
            self.notify("Ralph stopped")
            self.dismiss(None)
        
        # ═══════════════════════════════════════════════════════════
        # COMPLETE / STOPPED MODE
        # ═══════════════════════════════════════════════════════════
        elif btn == "btn-view-log":
            self.dismiss({"action": "view_log"})
        
        elif btn == "btn-resume":
            # Resume without adding task
            self.dismiss({"action": "resume", **self._get_settings()})
        
        elif btn == "btn-add-resume":
            # Add task (optional) and resume
            task = self.query_one("#task-input", TextArea).text.strip()
            priority = "normal"
            try:
                priority = self.query_one("#priority-select", Select).value
            except Exception:
                pass
            self.dismiss({
                "action": "add_resume",
                "task": task if task else None,
                "priority": priority,
                **self._get_settings()
            })
        
        elif btn == "btn-new-mission":
            # New mission (reset) - requires task
            task = self.query_one("#task-input", TextArea).text.strip()
            if not task:
                self.notify("Enter new mission description", severity="warning")
                return
            self.dismiss({"action": "new_mission", "task": task, **self._get_settings()})
        
        # ═══════════════════════════════════════════════════════════
        # NEW MODE
        # ═══════════════════════════════════════════════════════════
        elif btn == "btn-ralph-start":
            task = self.query_one("#task-input", TextArea).text.strip()
            if not task:
                self.notify("Enter task description", severity="warning")
                return
            self.dismiss({"action": "start", "task": task, **self._get_settings()})



class RalphMonitorScreen(Screen):
    """Screen for monitoring Ralph progress with 3-panel layout."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("p", "pause", "Pause", priority=True),
        Binding("s", "stop", "Stop", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("f", "force_restart", "Force Restart", priority=True),
        Binding("tab", "switch_panel", "Switch Panel", priority=True),
    ]
    
    CSS = """
    #ralph-monitor {
        height: 100%;
    }
    #ralph-header {
        height: auto;
        padding: 0 1;
        background: $surface;
    }
    #ralph-panels {
        height: 1fr;
    }
    #left-panel {
        width: 25%;
        border: solid green;
    }
    #right-panel {
        width: 75%;
    }
    #plan-panel {
        height: 40%;
        border: solid cyan;
    }
    #output-panel {
        height: 60%;
        border: solid blue;
    }
    .panel-title {
        text-style: bold;
        background: $surface;
        padding: 0 1;
        height: 1;
    }
    #iterations-tree {
        height: 1fr;
    }
    #plan-log {
        height: 1fr;
    }
    #output-log {
        height: 1fr;
    }
    """
    
    def __init__(self, project: Project, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.ralph = RalphManager(project)
        self._refresh_timer = None
        self._selected_iteration = None
        self._selected_filepath = None
        self._known_iterations = set()
        self._last_prd_mtime = 0
        self._last_iter_mtime = 0
        self._last_live_mtime = 0
        self._live_mode = True  # True = show live log, False = show specific iteration
        self._last_output_hash = ""
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="ralph-monitor"):
            with Container(id="ralph-header"):
                yield Label(f"* Ralph: {self.project.name}", id="ralph-title")
                yield Static("Status: Loading...", id="ralph-status")
                yield ProgressBar(total=100, id="ralph-progress")
                yield Static("Current Task: ...", id="ralph-task")
            
            with Horizontal(id="ralph-panels"):
                with Container(id="left-panel"):
                    yield Label("Iterations", classes="panel-title")
                    yield Tree("Iterations", id="iterations-tree")
                
                with Vertical(id="right-panel"):
                    with Container(id="plan-panel"):
                        yield Label("Plan", classes="panel-title", id="plan-title")
                        yield RichLog(id="plan-log", highlight=True, max_lines=500, markup=True, wrap=True)
                    
                    with Container(id="output-panel"):
                        yield Label("Output", classes="panel-title", id="output-title")
                        yield RichLog(id="output-log", highlight=True, max_lines=2000, markup=True, wrap=True)
        yield Footer()
    
    def on_mount(self):
        self._refresh_timer = self.set_interval(2.0, self.refresh_all)
        self.refresh_status()
        self.load_iterations()
        self._show_plan()
        self._show_live_output()  # Start in live mode
    
    def on_unmount(self):
        if self._refresh_timer:
            self._refresh_timer.stop()
    
    def refresh_all(self):
        """Refresh status and output."""
        self.refresh_status()
        self.check_new_iterations()
        self._check_plan_updated()
        
        if self._live_mode:
            self._update_live_output()
        else:
            self._update_iteration_output()
    
    def refresh_status(self):
        config = self.ralph.get_config()
        done, total = self.ralph.get_progress()
        
        status = config.get("status", "unknown")
        iteration = config.get("currentIteration", 0)
        max_iter = config.get("maxIterations", "?")
        
        # Check if process is actually running
        process_running = self._is_process_running()
        
        if status == "running" and not process_running:
            status_display = "[yellow]STALE[/] (process dead, press F to restart)"
            # Note: Don't auto-fix here - let Resume handle it
            # The _resume_ralph function will detect and fix zombie status
        elif status == "running" and process_running:
            status_display = f"[green]{status}[/]"
        elif status == "starting":
            status_display = f"[cyan]{status}[/] (daemon initializing...)"
        elif status == "error":
            status_display = f"[red]{status}[/] (press F to restart)"
        elif status in ["stopped", "paused"]:
            status_display = f"[yellow]{status}[/]"
        else:
            status_display = status
        
        self.query_one("#ralph-status", Static).update(
            f"Status: {status_display} | Iteration: {iteration}/{max_iter}"
        )
        
        if total > 0:
            progress = int((done / total) * 100)
            self.query_one("#ralph-progress", ProgressBar).update(progress=progress)
        
        task = self.ralph.get_current_task()
        if task:
            self.query_one("#ralph-task", Static).update(
                f"Current Task: {task.get('id', '?')} - {task.get('title', '?')}"
            )
        else:
            prd = self.ralph.get_prd()
            if prd.get("verified"):
                self.query_one("#ralph-task", Static).update("Project VERIFIED and complete!")
            elif status in ["complete", "finished"]:
                self.query_one("#ralph-task", Static).update("All tasks complete")
            else:
                self.query_one("#ralph-task", Static).update("Current Task: ...")
    
    def _is_process_running(self) -> bool:
        """Check if Ralph daemon is running in container."""
        try:
            return Docker.is_ralph_daemon_running(self.project)
        except Exception:
            return False
    
    def load_iterations(self):
        """Load all iterations into tree."""
        tree = self.query_one("#iterations-tree", Tree)
        tree.clear()
        tree.root.expand()
        self._known_iterations.clear()
        
        iterations_dir = self.ralph.ralph_dir / "iterations"
        if not iterations_dir.exists():
            tree.root.add_leaf("(no iterations yet)")
            return
        
        files = sorted(iterations_dir.glob("iteration_*.json"))
        
        for f in files:
            try:
                data = json.loads(f.read_text())
                num = data.get("iteration", 0)
                iter_type = data.get("type", "?")
                result = data.get("result", "in_progress")
                
                icon = self._get_status_icon(result)
                type_label = self._get_iter_type_label(iter_type)
                label = f"{icon} [{num}] {type_label}: {result[:20]}"
                tree.root.add_leaf(label, data={"file": str(f), "iteration": num, "result": result})
                self._known_iterations.add(num)
                
            except Exception:
                pass
        
        if files:
            # Select the latest iteration automatically
            try:
                latest = files[-1]
                data = json.loads(latest.read_text())
                self._selected_iteration = data.get("iteration", 0)
                self.show_iteration_detail(str(latest))
            except Exception:
                pass
    
    def check_new_iterations(self):
        """Check for new iteration files and update existing ones."""
        iterations_dir = self.ralph.ralph_dir / "iterations"
        if not iterations_dir.exists():
            return
        
        files = sorted(iterations_dir.glob("iteration_*.json"))
        tree = self.query_one("#iterations-tree", Tree)
        
        # Track if we need to reload the tree (for status updates)
        needs_reload = False
        
        for f in files:
            try:
                data = json.loads(f.read_text())
                num = data.get("iteration", 0)
                iter_type = data.get("type", "?")
                result = data.get("result", "in_progress")
                
                if num not in self._known_iterations:
                    # New iteration - add to tree
                    icon = self._get_status_icon(result)
                    type_label = self._get_iter_type_label(iter_type)
                    label = f"{icon} [{num}] {type_label}: {result[:20]}"
                    tree.root.add_leaf(label, data={"file": str(f), "iteration": num, "result": result})
                    self._known_iterations.add(num)
                    
                    self._selected_iteration = num
                    self.show_iteration_detail(str(f))
                else:
                    # Existing iteration - check if result changed
                    for child in tree.root.children:
                        if child.data and child.data.get("iteration") == num:
                            old_result = child.data.get("result", "in_progress")
                            if old_result != result:
                                # Result changed - need to reload tree
                                needs_reload = True
                                break
                    if needs_reload:
                        break
                    
            except Exception:
                pass
        
        # Reload tree if any status changed
        if needs_reload:
            self.load_iterations()
    
    def _get_status_icon(self, result: str) -> str:
        """Get icon for iteration status."""
        if result in ["in_progress", "running..."]:
            return "[yellow]►[/]"
        elif "DONE" in result or result == "VERIFIED" or result == "ARCHITECT_DONE":
            return "[green]✓[/]"
        elif result in ["UNKNOWN", "STUCK"]:
            return "[red]✗[/]"
        else:
            return "[yellow]?[/]"
    
    def _get_iter_type_label(self, iter_type: str) -> str:
        """Get human-readable label for iteration type."""
        labels = {
            "planning": "📋 Initial Planning",
            "add_planning": "➕ Add Tasks",
            "architect": "🏗️ Architect Review",
            "worker": "⚙️ Worker",
            "verification": "✅ Verify",
        }
        return labels.get(iter_type, iter_type)
    
    def _check_plan_updated(self):
        """Check if prd.json changed."""
        try:
            if self.ralph.prd_file.exists():
                current_mtime = self.ralph.prd_file.stat().st_mtime
                if current_mtime > self._last_prd_mtime:
                    self._last_prd_mtime = current_mtime
                    self._show_plan()
        except Exception:
            pass
    
    def _show_live_output(self):
        """Show live/combined output."""
        self._live_mode = True
        self._last_output_hash = ""
        output_title = self.query_one("#output-title", Label)
        output_title.update("Output | [green]● LIVE[/]")
        self._update_live_output()
    
    def _update_live_output(self):
        """Update live output from latest iteration."""
        iterations_dir = self.ralph.ralph_dir / "iterations"
        
        output = ""
        source = ""
        
        if not iterations_dir.exists():
            return
        
        # Find latest iteration files
        json_files = list(iterations_dir.glob("iteration_*.json"))
        log_files = list(iterations_dir.glob("iteration_*.log"))
        
        if not json_files and not log_files:
            return
        
        # Get latest by modification time
        latest_json = max(json_files, key=lambda f: f.stat().st_mtime) if json_files else None
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime) if log_files else None
        
        # Determine which is newer
        json_mtime = latest_json.stat().st_mtime if latest_json else 0
        log_mtime = latest_log.stat().st_mtime if latest_log else 0
        latest_mtime = max(json_mtime, log_mtime)
        
        # Only update if changed
        if latest_mtime <= self._last_live_mtime:
            return
        
        self._last_live_mtime = latest_mtime
        
        # Strategy: prefer parsed JSON, fall back to parsing log
        if latest_json and json_mtime >= log_mtime - 5:  # JSON is up to date (within 5s)
            output, source = self._read_iteration_json(latest_json)
        elif latest_log:
            output, source = self._parse_live_log(latest_log)
        
        if source:
            output_title = self.query_one("#output-title", Label)
            is_active = (time.time() - latest_mtime) < 15
            status = "[green]● LIVE[/]" if is_active else "[yellow]○ idle[/]"
            output_title.update(f"Output | {status} {source}")
        
        self._render_output(output, is_live=True)
    
    def _read_iteration_json(self, json_file: Path) -> Tuple[str, str]:
        """Read parsed assistant messages from iteration JSON."""
        try:
            data = json.loads(json_file.read_text())
            messages = data.get("assistant_messages", [])
            
            if messages:
                # Join messages with clear separators
                output = '\n\n'.join(messages)
                return output, f"{json_file.stem} (parsed)"
            
            return "", ""
        except Exception:
            return "", ""
    
    def _parse_live_log(self, log_file: Path) -> Tuple[str, str]:
        """Parse JSON events from log file to extract assistant messages."""
        try:
            content = log_file.read_text()
            
            # Use shared parser
            events, messages = parse_openhands_json_events(content)
            
            if messages:
                return '\n\n'.join(messages), f"{log_file.stem} (live)"
            
            # No JSON found - return cleaned raw text
            return clean_openhands_output(content), f"{log_file.stem} (raw)"
            
        except Exception as e:
            return f"Error reading log: {e}", ""
    
    def _update_iteration_output(self):
        """Update output for selected iteration."""
        if not self._selected_filepath:
            return
        
        try:
            fp = Path(self._selected_filepath)
            if not fp.exists():
                return
            
            mtime = fp.stat().st_mtime
            if mtime <= self._last_iter_mtime:
                return  # No change
            
            self._last_iter_mtime = mtime
            
            # Primary: read from JSON (has parsed assistant_messages)
            output, _ = self._read_iteration_json(fp)
            
            # Fallback: parse log file
            if not output:
                log_file = fp.with_suffix('.log')
                if log_file.exists():
                    output, _ = self._parse_live_log(log_file)
            
            self._render_output(output, is_live=False)
            
        except Exception:
            pass
    
    def show_iteration_detail(self, filepath: str):
        """Show details of selected iteration."""
        self._live_mode = False
        self._last_output_hash = ""
        
        try:
            fp = Path(filepath)
            data = json.loads(fp.read_text())
            
            self._selected_iteration = data.get("iteration", 0)
            self._selected_filepath = filepath
            self._last_iter_mtime = 0  # Force refresh
            
            iter_type = data.get("type", "?")
            result = data.get("result", "running...")
            elapsed = data.get("elapsed_seconds", 0)
            
            output_title = self.query_one("#output-title", Label)
            output_title.update(f"Output | Iter {self._selected_iteration} ({iter_type}) [{elapsed}s]")
            
            self._update_iteration_output()
            
        except Exception as e:
            output_log = self.query_one("#output-log", RichLog)
            output_log.clear()
            output_log.write(f"Error: {e}")
    
    def _show_plan(self):
        """Show current plan in plan panel."""
        plan_log = self.query_one("#plan-log", RichLog)
        plan_log.clear()
        
        try:
            prd = self.ralph.get_prd()
            stories = prd.get("userStories", [])
            phase = prd.get("phase", "planning")
            
            plan_log.write(f"Phase: {phase} | Total: {len(stories)} tasks")
            plan_log.write("")
            
            if not stories:
                plan_log.write("(no tasks yet - planning in progress...)")
                return
            
            done = [s for s in stories if s.get("passes")]
            pending = [s for s in stories if not s.get("passes")]
            
            if pending:
                plan_log.write(f"PENDING ({len(pending)}):")
                for s in sorted(pending, key=lambda x: x.get("priority", 999)):
                    task_id = s.get("id", "?")
                    title = s.get("title", "Untitled")
                    priority = s.get("priority", "?")
                    deps = s.get("dependsOn", [])
                    deps_str = f" [deps: {', '.join(deps)}]" if deps else ""
                    plan_log.write(f"  [{task_id}] P{priority}: {title}{deps_str}")
                plan_log.write("")
            
            if done:
                plan_log.write(f"COMPLETED ({len(done)}):")
                for s in done:
                    task_id = s.get("id", "?")
                    title = s.get("title", "Untitled")
                    plan_log.write(f"  [{task_id}] {title}")
                
        except Exception as e:
            plan_log.write(f"(error loading plan: {e})")
    
    def _render_output(self, output: str, is_live: bool = False):
        """Render output as formatted Markdown using Rich."""
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.text import Text
        
        output_log = self.query_one("#output-log", RichLog)
        
        if not output or not output.strip():
            if self._last_output_hash != "empty":
                output_log.clear()
                output_log.write("[dim]Waiting for output...[/]")
                self._last_output_hash = "empty"
            return
        
        # Check if changed (before expensive formatting)
        output_hash = str(hash(output))
        if output_hash == self._last_output_hash:
            return
        
        self._last_output_hash = output_hash
        output_log.clear()
        
        # Truncate for live mode to keep UI responsive
        if is_live and len(output) > 15000:
            output = f"... (truncated {len(output) - 15000} chars) ...\n\n" + output[-15000:]
        
        # Render as Markdown - Rich handles all formatting elegantly
        try:
            md = Markdown(output)
            output_log.write(md)
        except Exception:
            # Fallback to plain text if Markdown fails
            output_log.write(output)
    
    def on_tree_node_selected(self, event: Tree.NodeSelected):
        """Handle iteration selection from tree."""
        # Click on root "Iterations" = live mode
        if event.node.is_root:
            self._show_live_output()
            return
        
        # Click on specific iteration = show its log
        if event.node.data and "file" in event.node.data:
            self.show_iteration_detail(event.node.data["file"])
    
    def action_go_back(self):
        self.app.pop_screen()
    
    def action_pause(self):
        self.ralph.request_pause()
        self.notify("Pause requested - state saved")
    
    def action_stop(self):
        self.ralph.request_stop()
        self.notify("Stop requested - state saved")
    
    def action_refresh(self):
        """Manual refresh."""
        self.refresh_status()
        self._known_iterations.clear()
        self.load_iterations()
        self._show_plan()
        self.notify("Refreshed")
    
    def action_force_restart(self):
        """Force restart Ralph - clean up and restart process."""
        def do_restart(confirmed):
            if confirmed:
                self._do_force_restart()
        
        self.app.push_screen(
            ConfirmScreen("Force restart Ralph? This will kill any running process and restart."),
            do_restart
        )
    
    def _do_force_restart(self):
        """Actually perform the force restart - stops and starts daemon in container."""
        try:
            # Stop daemon in container
            Docker.stop_ralph_daemon(self.project)
            
            # Clear lock files via Docker
            Docker.exec_in_container(
                self.project.container_name,
                "rm -f /workspace/.ralph/.ralph.lock /workspace/.ralph/.run.lock",
                timeout=5
            )
            
            # Update status
            self.ralph.update_config("status", "stopped")
            self.ralph.update_config("pauseRequested", False)
            
            time.sleep(0.5)
            
            # Ensure container is running
            if not Docker.container_running(self.project.container_name):
                if not Docker.ensure_container_running(self.project):
                    raise Exception("Failed to start container")
            
            # Setup and start daemon
            if not Docker.setup_ralph_daemon(self.project):
                raise Exception("Failed to setup daemon")
            
            if not Docker.start_ralph_daemon(self.project):
                raise Exception("Failed to start daemon")
            
            self.ralph.update_config("status", "running")
            self.ralph.add_progress_entry(
                self.ralph.get_config().get("currentIteration", 0),
                "Force restarted by user"
            )
            
            self.notify("Ralph force restarted!")
            self.refresh_status()
            
        except Exception as e:
            self.notify(f"Force restart failed: {e}", severity="error")
    
    def action_switch_panel(self):
        """Switch focus between panels."""
        try:
            tree = self.query_one("#iterations-tree", Tree)
            plan_log = self.query_one("#plan-log", RichLog)
            output_log = self.query_one("#output-log", RichLog)
            
            if tree.has_focus:
                plan_log.focus()
            elif plan_log.has_focus:
                output_log.focus()
            else:
                tree.focus()
        except Exception:
            pass


class StartupLogScreen(Screen):
    """Screen showing startup logs in real-time."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
    ]
    
    CSS = """
    #startup-log-container {
        height: 100%;
        padding: 1;
    }
    #startup-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    #log-area {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        overflow-y: auto;
    }
    #status-line {
        height: auto;
        margin-top: 1;
        text-align: center;
    }
    """
    
    def __init__(self, project: Project, mode: str = "ralph", config_data: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.mode = mode  # "ralph" or "session"
        self.config_data = config_data or {}
        self._logs: list[str] = []
        self._success = False
        self._finished = False
        self._error_msg = ""
    
    def compose(self) -> ComposeResult:
        title = "Starting Ralph..." if self.mode == "ralph" else "Starting Session..."
        with Container(id="startup-log-container"):
            yield Static(f"[bold]{title}[/]", id="startup-title", markup=True)
            yield Static("", id="log-area")
            yield Static("[dim]Initializing...[/]", id="status-line", markup=True)
    
    def on_mount(self):
        """Start the operation when screen is mounted."""
        ralph_mode = self.config_data.get("mode", "start")
        
        if self.mode == "ralph":
            if ralph_mode == "resume":
                self._resume_ralph_with_logs()
            elif ralph_mode == "new_mission":
                self._new_mission_ralph_with_logs()
            else:
                self._start_ralph_with_logs()
        else:
            self._start_session_with_logs()
    
    def add_log(self, message: str, level: str = "info"):
        """Add a log message to the display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level == "error":
            formatted = f"[red]✗ [{timestamp}] {message}[/]"
        elif level == "success":
            formatted = f"[green]✓ [{timestamp}] {message}[/]"
        elif level == "warning":
            formatted = f"[yellow]! [{timestamp}] {message}[/]"
        else:
            formatted = f"[dim][{timestamp}][/] {message}"
        
        self._logs.append(formatted)
        
        # Update display
        log_text = "\n".join(self._logs[-30:])  # Show last 30 lines
        try:
            self.query_one("#log-area", Static).update(log_text)
        except Exception:
            pass
    
    def set_status(self, message: str, is_error: bool = False):
        """Update status line."""
        try:
            if is_error:
                self.query_one("#status-line", Static).update(f"[red bold]{message}[/]")
            else:
                self.query_one("#status-line", Static).update(f"[cyan]{message}[/]")
        except Exception:
            pass
    
    def action_cancel(self):
        """Cancel and go back."""
        if self._finished:
            self.dismiss({"success": self._success, "error": self._error_msg})
        else:
            self.add_log("Cancelled by user", "warning")
            self.dismiss({"success": False, "error": "Cancelled"})
    
    def _show_daemon_failure(self, p: Project, ralph, error_msg: str):
        """Show daemon failure with log output."""
        self.app.call_from_thread(self.add_log, f"{error_msg}!", "error")
        
        _, log_output = Docker.exec_in_container(
            p.container_name,
            "tail -15 /workspace/.ralph/ralph_daemon.log 2>/dev/null || echo '(no log)'",
            timeout=5
        )
        self.app.call_from_thread(self.add_log, "--- Daemon Log ---", "warning")
        for line in log_output.strip().split("\n")[:10]:
            self.app.call_from_thread(self.add_log, f"  {line}")
        
        self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
        ralph.update_config("status", "stopped")
        self._error_msg = error_msg
        self._finished = True
    
    @work(thread=True)
    def _start_ralph_with_logs(self):
        """Start Ralph with detailed logging."""
        p = self.project
        config_data = self.config_data
        task = config_data.get("task", "")
        
        try:
            self.app.call_from_thread(self.add_log, f"Project: {p.name}")
            self.app.call_from_thread(self.add_log, f"Task: {task[:50]}...")
            self.app.call_from_thread(self.set_status, "Creating Ralph manager...")
            
            ralph = RalphManager(p)
            config = RalphConfig(
                task_description=task,
                max_iterations=config_data.get("max_iterations", 0),
                architect_interval=config_data.get("architect_interval", 10)
            )
            
            self.app.call_from_thread(self.add_log, "Initializing Ralph structure...")
            self.app.call_from_thread(self.set_status, "Initializing structure...")
            
            if not ralph.init_structure(config):
                self.app.call_from_thread(self.add_log, "Failed to initialize structure!", "error")
                self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                self._error_msg = "Failed to initialize Ralph structure"
                self._finished = True
                return
            
            self.app.call_from_thread(self.add_log, "Structure initialized", "success")
            
            # Update config
            ralph.update_config("status", "starting")
            ralph.update_config("condenseInterval", config_data.get("condense_interval", 15))
            
            # Check container
            self.app.call_from_thread(self.add_log, f"Checking container: {p.container_name}")
            self.app.call_from_thread(self.set_status, "Checking Docker container...")
            
            if not Docker.container_running(p.container_name):
                self.app.call_from_thread(self.add_log, "Container not running, starting...", "warning")
                self.app.call_from_thread(self.set_status, "Starting Docker container...")
                
                if not Docker.ensure_container_running(p):
                    self.app.call_from_thread(self.add_log, "Failed to start container!", "error")
                    self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                    ralph.update_config("status", "stopped")
                    self._error_msg = "Failed to start Docker container"
                    self._finished = True
                    return
                
                self.app.call_from_thread(self.add_log, "Container started", "success")
            else:
                self.app.call_from_thread(self.add_log, "Container already running", "success")
            
            # Setup daemon (includes dependency installation)
            self.app.call_from_thread(self.add_log, "Setting up Ralph daemon...")
            self.app.call_from_thread(self.set_status, "Copying daemon files...")
            
            # Check if dependencies need installation
            code, dep_check = Docker.exec_in_container(
                p.container_name,
                "python3 -c 'import sentence_transformers' 2>/dev/null && echo 'OK' || echo 'NEED'",
                timeout=10
            )
            if "NEED" in dep_check:
                self.app.call_from_thread(self.add_log, "Dependencies missing, installing (2-5 min)...", "warning")
                self.app.call_from_thread(self.set_status, "Installing sentence-transformers (~500MB)...")
            
            if not Docker.setup_ralph_daemon(p):
                self.app.call_from_thread(self.add_log, "Failed to setup daemon!", "error")
                self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                ralph.update_config("status", "stopped")
                self._error_msg = "Failed to setup Ralph daemon"
                self._finished = True
                return
            
            self.app.call_from_thread(self.add_log, "Daemon setup complete", "success")
            
            # Start daemon
            self.app.call_from_thread(self.add_log, "Starting Ralph daemon...")
            self.app.call_from_thread(self.set_status, "Starting daemon...")
            
            success = Docker.start_ralph_daemon(p)
            
            if success:
                # Verify daemon is still running after brief wait
                time.sleep(2)
                if Docker.is_ralph_daemon_running(p):
                    ralph.update_config("status", "running")
                    self.app.call_from_thread(self.add_log, "Ralph daemon running!", "success")
                    self.app.call_from_thread(self.set_status, "SUCCESS - Opening monitor...")
                    self._success = True
                    self._finished = True
                    time.sleep(1)
                    self.app.call_from_thread(self.dismiss, {"success": True, "error": ""})
                else:
                    self._show_daemon_failure(p, ralph, "Daemon crashed after starting")
            else:
                self._show_daemon_failure(p, ralph, "Daemon failed to start")
            
        except Exception as e:
            self.app.call_from_thread(self.add_log, f"Exception: {e}", "error")
            self.app.call_from_thread(self.add_log, traceback.format_exc()[:200])
            self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
            self._error_msg = str(e)
            self._finished = True
    
    @work(thread=True)
    def _start_session_with_logs(self):
        """Start regular session with detailed logging."""
        p = self.project
        config_data = self.config_data
        
        try:
            self.app.call_from_thread(self.add_log, f"Project: {p.name}")
            self.app.call_from_thread(self.set_status, "Checking Docker...")
            
            # Check container
            self.app.call_from_thread(self.add_log, f"Container: {p.container_name}")
            
            if not Docker.container_running(p.container_name):
                self.app.call_from_thread(self.add_log, "Starting container...", "warning")
                self.app.call_from_thread(self.set_status, "Starting Docker container...")
                
                if not Docker.ensure_container_running(p):
                    self.app.call_from_thread(self.add_log, "Failed to start container!", "error")
                    self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                    self._error_msg = "Failed to start container"
                    self._finished = True
                    return
                
                self.app.call_from_thread(self.add_log, "Container started", "success")
            else:
                self.app.call_from_thread(self.add_log, "Container running", "success")
            
            # Session setup
            session_id = config_data.get("session_id")
            
            if session_id:
                self.app.call_from_thread(self.add_log, f"Resuming session: {session_id[:8]}...")
            else:
                self.app.call_from_thread(self.add_log, "Creating new session...")
            
            self.app.call_from_thread(self.set_status, "Starting tmux session...")
            
            # Start background session
            self.app.call_from_thread(self.add_log, "Starting OpenHands in background...")
            
            if Docker.start_background_session(p, session_id):
                self.app.call_from_thread(self.add_log, "Session started!", "success")
                self.app.call_from_thread(self.set_status, "SUCCESS - Attaching to session...")
                self._success = True
                self._finished = True
                
                time.sleep(1)
                self.app.call_from_thread(self.dismiss, {"success": True, "error": ""})
            else:
                self.app.call_from_thread(self.add_log, "Failed to start session!", "error")
                self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                self._error_msg = "Failed to start background session"
                self._finished = True
            
        except Exception as e:
            self.app.call_from_thread(self.add_log, f"Exception: {e}", "error")
            self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
            self._error_msg = str(e)
            self._finished = True
    
    @work(thread=True)
    def _resume_ralph_with_logs(self):
        """Resume Ralph with detailed logging."""
        p = self.project
        config_data = self.config_data
        
        try:
            self.app.call_from_thread(self.add_log, f"Resuming project: {p.name}")
            self.app.call_from_thread(self.set_status, "Loading Ralph state...")
            
            ralph = RalphManager(p)
            
            # Check for zombie status
            current_status = ralph.get_config().get("status", "unknown")
            self.app.call_from_thread(self.add_log, f"Current status: {current_status}")
            
            if current_status == "running":
                process_running = Docker.is_ralph_daemon_running(p)
                if not process_running:
                    self.app.call_from_thread(self.add_log, "Status was 'running' but daemon dead - resetting", "warning")
                    ralph.update_config("status", "paused")
            
            # Update config
            ralph.update_config("status", "starting")
            ralph.update_config("maxIterations", config_data.get("max_iterations", 0))
            ralph.update_config("architectInterval", config_data.get("architect_interval", 10))
            ralph.update_config("condenseInterval", config_data.get("condense_interval", 15))
            ralph.update_config("pauseRequested", False)
            
            self.app.call_from_thread(self.add_log, "Config updated", "success")
            
            # Process pending tasks
            pending_count = ralph.process_pending_tasks()
            if pending_count > 0:
                self.app.call_from_thread(self.add_log, f"Added {pending_count} pending tasks", "success")
            
            prd = ralph.get_prd()
            prd["verified"] = False
            ralph.save_prd(prd)
            
            ralph.add_progress_entry(0, "Resumed - continuing from last state")
            
            # Start daemon (same as _start_ralph_with_logs from here)
            self.app.call_from_thread(self.add_log, f"Checking container: {p.container_name}")
            self.app.call_from_thread(self.set_status, "Checking Docker container...")
            
            if not Docker.container_running(p.container_name):
                self.app.call_from_thread(self.add_log, "Container not running, starting...", "warning")
                self.app.call_from_thread(self.set_status, "Starting Docker container...")
                
                if not Docker.ensure_container_running(p):
                    self.app.call_from_thread(self.add_log, "Failed to start container!", "error")
                    self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                    ralph.update_config("status", "stopped")
                    self._error_msg = "Failed to start Docker container"
                    self._finished = True
                    return
                
                self.app.call_from_thread(self.add_log, "Container started", "success")
            else:
                self.app.call_from_thread(self.add_log, "Container already running", "success")
            
            # Setup daemon
            self.app.call_from_thread(self.add_log, "Setting up Ralph daemon...")
            self.app.call_from_thread(self.set_status, "Setting up daemon files...")
            
            if not Docker.setup_ralph_daemon(p):
                self.app.call_from_thread(self.add_log, "Failed to setup daemon!", "error")
                self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                ralph.update_config("status", "stopped")
                self._error_msg = "Failed to setup Ralph daemon"
                self._finished = True
                return
            
            self.app.call_from_thread(self.add_log, "Daemon files ready", "success")
            
            # Start daemon
            self.app.call_from_thread(self.add_log, "Starting Ralph daemon...")
            self.app.call_from_thread(self.set_status, "Starting daemon...")
            
            success = Docker.start_ralph_daemon(p)
            
            if success:
                self.app.call_from_thread(self.add_log, "Daemon started, verifying...", "success")
                time.sleep(3)
                
                still_running = Docker.is_ralph_daemon_running(p)
                if still_running:
                    ralph.update_config("status", "running")
                    self.app.call_from_thread(self.add_log, "Ralph daemon verified!", "success")
                    self.app.call_from_thread(self.set_status, "SUCCESS - Opening monitor...")
                    self._success = True
                    self._finished = True
                    time.sleep(1)
                    self.app.call_from_thread(self.dismiss, {"success": True, "error": ""})
                else:
                    self.app.call_from_thread(self.add_log, "Daemon died after starting!", "error")
                    _, log_output = Docker.exec_in_container(
                        p.container_name,
                        "tail -30 /workspace/.ralph/ralph_daemon.log 2>/dev/null || echo 'No log'",
                        timeout=5
                    )
                    for line in log_output.strip().split("\n")[:15]:
                        self.app.call_from_thread(self.add_log, f"  {line}")
                    self.app.call_from_thread(self.set_status, "FAILED - Daemon crashed", True)
                    ralph.update_config("status", "stopped")
                    self._error_msg = "Daemon crashed"
                    self._finished = True
            else:
                _, log_output = Docker.exec_in_container(
                    p.container_name,
                    "tail -20 /workspace/.ralph/ralph_daemon.log 2>/dev/null || echo 'No log'",
                    timeout=5
                )
                
                self.app.call_from_thread(self.add_log, "Daemon failed to start!", "error")
                for line in log_output.strip().split("\n")[:10]:
                    self.app.call_from_thread(self.add_log, f"  {line}")
                
                self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                ralph.update_config("status", "stopped")
                self._error_msg = "Daemon failed to start"
                self._finished = True
            
        except Exception as e:
            self.app.call_from_thread(self.add_log, f"Exception: {e}", "error")
            self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
            self._error_msg = str(e)
            self._finished = True
    
    @work(thread=True)
    def _new_mission_ralph_with_logs(self):
        """Start new mission with detailed logging."""
        p = self.project
        config_data = self.config_data
        task = config_data.get("task", "")
        
        try:
            self.app.call_from_thread(self.add_log, f"New mission for: {p.name}")
            self.app.call_from_thread(self.add_log, f"Task: {task[:50]}...")
            self.app.call_from_thread(self.set_status, "Preparing new mission...")
            
            ralph = RalphManager(p)
            
            # Save existing knowledge
            self.app.call_from_thread(self.add_log, "Preserving existing knowledge...")
            old_learnings = ralph._read_file("LEARNINGS.md") or ""
            old_architecture = ralph._read_file("ARCHITECTURE.md") or ""
            
            if old_learnings:
                self.app.call_from_thread(self.add_log, f"  LEARNINGS.md: {len(old_learnings)} bytes", "success")
            if old_architecture:
                self.app.call_from_thread(self.add_log, f"  ARCHITECTURE.md: {len(old_architecture)} bytes", "success")
            
            # Extract task references
            task_refs = re.findall(r'\[?(TASK-\d+)\]?[:\s]+([^\n\[\]]+)', task)
            
            # Create new PRD
            prd = {
                "projectName": p.name,
                "verified": False,
                "phase": "planning" if not task_refs else "execution",
                "taskDescription": task,
                "userStories": [{
                    "id": "PLAN",
                    "title": "Analyze and create execution plan",
                    "description": "Read the codebase, understand requirements, create detailed plan",
                    "type": "planning",
                    "priority": 0,
                    "passes": bool(task_refs),
                    "dependsOn": []
                }]
            }
            
            if task_refs:
                self.app.call_from_thread(self.add_log, f"Found {len(task_refs)} explicit tasks")
                for i, (task_id, task_title) in enumerate(task_refs):
                    task_title = task_title.strip()
                    if re.match(r'^P\d+:\s*', task_title):
                        task_title = re.sub(r'^P\d+:\s*', '', task_title)
                    prd["userStories"].append({
                        "id": task_id,
                        "title": task_title,
                        "description": f"Task from mission: {task_title}",
                        "type": "feature",
                        "priority": i + 1,
                        "passes": False,
                        "dependsOn": []
                    })
            
            ralph.save_prd(prd)
            self.app.call_from_thread(self.add_log, "PRD created", "success")
            
            # Reset config
            ralph.update_config("iteration", 0)
            ralph.update_config("currentIteration", 0)
            ralph.update_config("maxIterations", config_data.get("max_iterations", 0))
            ralph.update_config("architectInterval", config_data.get("architect_interval", 10))
            ralph.update_config("condenseInterval", config_data.get("condense_interval", 15))
            ralph.update_config("status", "starting")
            
            # Update mission file
            timestamp = datetime.now().isoformat()
            mission_content = f"# {task}\n\n*New mission started at {timestamp}*\n\n{task}"
            ralph._write_file("MISSION.md", mission_content)
            self.app.call_from_thread(self.add_log, "Mission file updated", "success")
            
            # Restore knowledge
            if old_learnings:
                ralph._write_file("LEARNINGS.md", old_learnings)
            if old_architecture:
                ralph._write_file("ARCHITECTURE.md", old_architecture)
            
            # Backup old iterations
            backup_subdir = timestamp.replace(":", "-")
            Docker.mkdir(ralph.container_name, f"{ralph.CONTAINER_RALPH_DIR}/iterations_backup/{backup_subdir}")
            Docker.exec_in_container(ralph.container_name, f"mv /workspace/.ralph/iterations/* /workspace/.ralph/iterations_backup/{backup_subdir}/ 2>/dev/null || true")
            self.app.call_from_thread(self.add_log, "Old iterations backed up", "success")
            
            ralph.add_progress_entry(0, f"New mission: {task[:50]}...")
            
            # Now start daemon (same as other methods)
            self.app.call_from_thread(self.add_log, f"Checking container: {p.container_name}")
            self.app.call_from_thread(self.set_status, "Checking Docker container...")
            
            if not Docker.container_running(p.container_name):
                self.app.call_from_thread(self.add_log, "Container not running, starting...", "warning")
                self.app.call_from_thread(self.set_status, "Starting Docker container...")
                
                if not Docker.ensure_container_running(p):
                    self.app.call_from_thread(self.add_log, "Failed to start container!", "error")
                    self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                    ralph.update_config("status", "stopped")
                    self._error_msg = "Failed to start Docker container"
                    self._finished = True
                    return
                
                self.app.call_from_thread(self.add_log, "Container started", "success")
            else:
                self.app.call_from_thread(self.add_log, "Container already running", "success")
            
            # Setup daemon
            self.app.call_from_thread(self.add_log, "Setting up Ralph daemon...")
            self.app.call_from_thread(self.set_status, "Setting up daemon files...")
            
            if not Docker.setup_ralph_daemon(p):
                self.app.call_from_thread(self.add_log, "Failed to setup daemon!", "error")
                self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                ralph.update_config("status", "stopped")
                self._error_msg = "Failed to setup Ralph daemon"
                self._finished = True
                return
            
            self.app.call_from_thread(self.add_log, "Daemon files ready", "success")
            
            # Start daemon
            self.app.call_from_thread(self.add_log, "Starting Ralph daemon...")
            self.app.call_from_thread(self.set_status, "Starting daemon...")
            
            success = Docker.start_ralph_daemon(p)
            
            if success:
                self.app.call_from_thread(self.add_log, "Daemon started, verifying...", "success")
                time.sleep(3)
                
                still_running = Docker.is_ralph_daemon_running(p)
                if still_running:
                    ralph.update_config("status", "running")
                    self.app.call_from_thread(self.add_log, "New mission running!", "success")
                    self.app.call_from_thread(self.set_status, "SUCCESS - Opening monitor...")
                    self._success = True
                    self._finished = True
                    time.sleep(1)
                    self.app.call_from_thread(self.dismiss, {"success": True, "error": ""})
                else:
                    self.app.call_from_thread(self.add_log, "Daemon died after starting!", "error")
                    _, log_output = Docker.exec_in_container(
                        p.container_name,
                        "tail -30 /workspace/.ralph/ralph_daemon.log 2>/dev/null || echo 'No log'",
                        timeout=5
                    )
                    for line in log_output.strip().split("\n")[:15]:
                        self.app.call_from_thread(self.add_log, f"  {line}")
                    self.app.call_from_thread(self.set_status, "FAILED - Daemon crashed", True)
                    ralph.update_config("status", "stopped")
                    self._error_msg = "Daemon crashed"
                    self._finished = True
            else:
                _, log_output = Docker.exec_in_container(
                    p.container_name,
                    "tail -20 /workspace/.ralph/ralph_daemon.log 2>/dev/null || echo 'No log'",
                    timeout=5
                )
                
                self.app.call_from_thread(self.add_log, "Daemon failed to start!", "error")
                for line in log_output.strip().split("\n")[:10]:
                    self.app.call_from_thread(self.add_log, f"  {line}")
                
                self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
                ralph.update_config("status", "stopped")
                self._error_msg = "Daemon failed to start"
                self._finished = True
            
        except Exception as e:
            self.app.call_from_thread(self.add_log, f"Exception: {e}", "error")
            self.app.call_from_thread(self.set_status, "FAILED - Press ESC to go back", True)
            self._error_msg = str(e)
            self._finished = True


class ConfirmScreen(ModalScreen):
    """Confirmation dialog."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self.message, id="confirm-message")
            with Horizontal(id="dialog-buttons"):
                yield Button("Yes", variant="error", id="btn-yes")
                yield Button("No", variant="primary", id="btn-no")
    
    def on_button_pressed(self, event: Button.Pressed):
        self.dismiss(event.button.id == "btn-yes")



class ContainersScreen(Screen):
    """Screen for managing containers."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "start_selected", "Start", priority=True),
        Binding("x", "stop_selected", "Stop", priority=True),
        Binding("d", "remove_selected", "Remove", priority=True),
        Binding("t", "restart_selected", "Restart", priority=True),
        Binding("h", "shell_selected", "Shell"),
    ]
    
    selected_container: reactive[Optional[str]] = reactive(None)
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Label("# Container Management", id="containers-title")
            yield Static("Select a container to manage it", id="container-hint")
            yield DataTable(id="containers-table", cursor_type="row")
            yield Static("Selected: none", id="selected-info")
            with Horizontal(id="container-actions-single"):
                yield Button("> Start", id="btn-container-start", variant="success")
                yield Button("X Stop", id="btn-container-stop", variant="warning")
                yield Button("T Restart", id="btn-container-restart", variant="primary")
                yield Button("D Remove", id="btn-container-remove", variant="error")
                yield Button("> Shell", id="btn-container-shell")
            with Horizontal(id="container-actions-all"):
                yield Button("Start All", id="btn-start-all")
                yield Button("Stop All", id="btn-stop-all")
                yield Button("Remove Stopped", id="btn-remove-stopped")
                yield Button("Back", id="btn-back")
        yield Footer()
    
    def on_mount(self):
        table = self.query_one("#containers-table", DataTable)
        table.cursor_type = "row"
        self.refresh_containers()
    
    def refresh_containers(self):
        table = self.query_one("#containers-table", DataTable)
        table.clear(columns=True)
        table.add_columns("", "Name", "State", "Size")
        
        containers = Docker.get_containers()
        for c in containers:
            is_running = c["status"].startswith("Up")
            status_icon = "[green]ON[/]" if is_running else "[dim]OFF[/]"
            state = "running" if is_running else "stopped"
            table.add_row(status_icon, c["name"], state, c["size"], key=c["name"])
        
        if not containers:
            self.query_one("#selected-info", Static).update("No containers found")
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        """Handle row selection."""
        if event.row_key:
            self.selected_container = str(event.row_key.value)
            self.query_one("#selected-info", Static).update(f"Selected: {self.selected_container}")
    
    def on_button_pressed(self, event: Button.Pressed):
        btn = event.button.id
        
        if btn == "btn-back":
            self.app.pop_screen()
        elif btn == "btn-container-start":
            self.action_start_selected()
        elif btn == "btn-container-stop":
            self.action_stop_selected()
        elif btn == "btn-container-restart":
            self.action_restart_selected()
        elif btn == "btn-container-remove":
            self.action_remove_selected()
        elif btn == "btn-container-shell":
            self.action_shell_selected()
        elif btn == "btn-start-all":
            self.start_all()
        elif btn == "btn-stop-all":
            self.stop_all()
        elif btn == "btn-remove-stopped":
            self.remove_stopped()
    
    def action_start_selected(self):
        if not self.selected_container:
            self.notify("Select a container first", severity="warning")
            return
        if Docker.start_container(self.selected_container):
            self.notify(f"Started: {self.selected_container}")
        else:
            self.notify(f"Failed to start: {self.selected_container}", severity="error")
        self.refresh_containers()
    
    def action_stop_selected(self):
        if not self.selected_container:
            self.notify("Select a container first", severity="warning")
            return
        if Docker.stop_container(self.selected_container):
            self.notify(f"Stopped: {self.selected_container}")
        else:
            self.notify(f"Failed to stop: {self.selected_container}", severity="error")
        self.refresh_containers()
    
    def action_restart_selected(self):
        """Restart container (stop + start). Triggers autostart scripts."""
        if not self.selected_container:
            self.notify("Select a container first", severity="warning")
            return
        
        self.notify(f"Restarting {self.selected_container}...")
        
        # Stop
        Docker.stop_container(self.selected_container)
        time.sleep(1)
        
        # Start
        if Docker.start_container(self.selected_container):
            self.notify(f"Restarted: {self.selected_container}")
        else:
            self.notify(f"Failed to restart: {self.selected_container}", severity="error")
        self.refresh_containers()
    
    def action_remove_selected(self):
        if not self.selected_container:
            self.notify("Select a container first", severity="warning")
            return
        Docker.stop_container(self.selected_container)
        if Docker.remove_container(self.selected_container, force=True):
            self.notify(f"Removed: {self.selected_container}")
            self.selected_container = None
            self.query_one("#selected-info", Static).update("Selected: none")
        else:
            self.notify(f"Failed to remove: {self.selected_container}", severity="error")
        self.refresh_containers()
    
    def action_shell_selected(self):
        """Open shell in selected container."""
        if not self.selected_container:
            self.notify("Select a container first", severity="warning")
            return
        
        if not Docker.container_running(self.selected_container):
            self.notify("Container is not running", severity="warning")
            return
        
        self.app.exit(result=("shell", self.selected_container))
    
    def start_all(self):
        containers = Docker.get_containers()
        count = 0
        for c in containers:
            if not c["status"].startswith("Up"):
                if Docker.start_container(c["name"]):
                    count += 1
        self.notify(f"Started {count} containers")
        self.refresh_containers()
    
    def stop_all(self):
        containers = Docker.get_containers()
        count = 0
        for c in containers:
            if c["status"].startswith("Up"):
                if Docker.stop_container(c["name"]):
                    count += 1
        self.notify(f"Stopped {count} containers")
        self.refresh_containers()
    
    def remove_stopped(self):
        containers = Docker.get_containers()
        count = 0
        for c in containers:
            if not c["status"].startswith("Up"):
                if Docker.remove_container(c["name"]):
                    count += 1
        self.notify(f"Removed {count} containers")
        self.refresh_containers()
    
    def action_go_back(self):
        self.app.pop_screen()
    
    def action_refresh(self):
        self.refresh_containers()


class ProjectSettingsScreen(ModalScreen):
    """Screen for project settings."""
    
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    
    def __init__(self, project: Project, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.qmd_mgr = QMDManager(project)
    
    def compose(self) -> ComposeResult:
        templates = ProjectManager.list_llm_templates() or ["(no templates)"]
        current_llm = self.project.get_llm_model()
        
        with Container(id="dialog"):
            yield Label(f"* Project Settings: {self.project.name}", id="dialog-title")
            
            yield Label("LLM Template:")
            yield Select([(t, t) for t in templates], id="llm-select", value=templates[0] if templates else None)
            yield Static(f"[dim]Current: {current_llm}[/]", markup=True)
            
            yield Label("")
            yield Static("[dim]QMD semantic search: enabled by default[/]", markup=True)
            
            yield Label("")
            yield Label("Actions:", classes="section-label")
            
            with Horizontal(classes="settings-row"):
                yield Button("Upgrade MCP", id="btn-update-mcp", variant="primary")
                yield Button("Update Skills", id="btn-update-skills")
            
            with Horizontal(classes="settings-row"):
                yield Button("Upgrade OpenHands", id="btn-upgrade-oh")
                yield Button("Pull Docker Image", id="btn-pull-image")
            
            with Horizontal(classes="settings-row"):
                yield Button("Manage Sessions", id="btn-sessions")
                yield Button("Force Restart MCP", id="btn-restart-gw", variant="warning")
            
            with Horizontal(classes="settings-row"):
                yield Button("Reset Container", id="btn-reset", variant="warning")
                yield Button("Delete Project", id="btn-delete", variant="error")
            
            with Horizontal(id="dialog-buttons"):
                yield Button("Apply LLM", variant="primary", id="btn-apply")
                yield Button("Close", id="btn-close")
    

    
    def on_button_pressed(self, event: Button.Pressed):
        btn = event.button.id
        
        if btn == "btn-close":
            self.dismiss(None)
        
        elif btn == "btn-apply":
            template = self.query_one("#llm-select", Select).value
            if template and template != "(no templates)":
                if ProjectManager.change_llm(self.project, template):
                    self.notify(f"LLM changed to: {template}. Restarting...")
                    self.dismiss({"action": "llm_changed"})
                else:
                    self.notify("Failed to change LLM", severity="error")
        
        elif btn == "btn-update-mcp":
            # Upgrade MCP: install new packages, remove old, restart gateways
            self.dismiss({"action": "upgrade_mcp"})
        
        elif btn == "btn-update-skills":
            if ProjectManager.update_skills(self.project):
                self.notify("Skills updated!")
            else:
                self.notify("Failed to update Skills", severity="error")
        
        elif btn == "btn-upgrade-oh":
            self.dismiss({"action": "upgrade_openhands"})
        
        elif btn == "btn-pull-image":
            self.dismiss({"action": "pull_image"})
        
        elif btn == "btn-sessions":
            self.dismiss({"action": "manage_sessions"})
        
        elif btn == "btn-restart-gw":
            self.dismiss({"action": "restart_gateways"})
        
        elif btn == "btn-reset":
            self.dismiss({"action": "reset"})
        
        elif btn == "btn-delete":
            self.dismiss({"action": "delete"})
    
    def action_cancel(self):
        self.dismiss(None)


class SessionSelectScreen(ModalScreen):
    """Screen for selecting session to resume."""
    
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    
    def __init__(self, project: Project, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.sessions = ProjectManager.get_sessions(project)
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(f"Resume Session: {self.project.name} ({len(self.sessions)} sessions)", id="dialog-title")
            yield ListView(id="session-list")
            with Horizontal(id="dialog-buttons"):
                yield Button("New Session", variant="primary", id="btn-new-session")
                yield Button("Cancel", id="btn-cancel")
    
    def on_mount(self):
        self.call_after_refresh(self._populate_list)
    
    def _populate_list(self):
        try:
            list_view = self.query_one("#session-list", ListView)
            for s in self.sessions:
                age = f"{s['age_days']}d ago" if s['age_days'] > 0 else "today"
                item = ListItem(Static(f"{s['id'][:12]}... ({age})"))
                item.session_id = s['id']
                list_view.append(item)
        except Exception:
            pass  # Widget not ready yet
    
    def on_list_view_selected(self, event: ListView.Selected):
        if event.item:
            session_id = getattr(event.item, 'session_id', None)
            if session_id:
                self.dismiss({"session": session_id})
    
    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-new-session":
            self.dismiss({"session": None})
    
    def action_cancel(self):
        self.dismiss(None)


class StartModeScreen(ModalScreen):
    """Screen for starting/viewing background session.
    
    All sessions run in background via tmux - you can close terminal
    and OpenHands keeps working. Use 'View' to attach to running session.
    """
    
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    
    def __init__(self, project: Project, session_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.session_id = session_id
        self.bg_running = Docker.is_background_running(project)
    
    def compose(self) -> ComposeResult:
        session_text = f"Session: {self.session_id[:12]}..." if self.session_id else "New session"
        
        with Container(id="dialog"):
            yield Label(f"Start: {self.project.name}", id="dialog-title")
            yield Static(f"[dim]{session_text}[/]", markup=True)
            yield Label("")
            
            if self.bg_running:
                yield Static("[green]● Background session is running[/]", markup=True)
                yield Static("[dim]Detach: Ctrl+B, D (keeps running)[/]", markup=True)
                yield Label("")
                with Horizontal(id="dialog-buttons"):
                    yield Button("View Session", variant="primary", id="btn-view")
                    yield Button("Stop", variant="warning", id="btn-stop")
                    yield Button("Cancel", id="btn-cancel")
            else:
                yield Static("[dim]Session runs in background (tmux)[/]", markup=True)
                yield Static("[dim]Close terminal - OpenHands keeps working[/]", markup=True)
                yield Static("[dim]Use 'View' to attach and interact[/]", markup=True)
                yield Label("")
                with Horizontal(id="dialog-buttons"):
                    yield Button("Start", variant="success", id="btn-start")
                    yield Button("Cancel", id="btn-cancel")
    
    def on_button_pressed(self, event: Button.Pressed):
        btn = event.button.id
        
        if btn == "btn-cancel":
            self.dismiss(None)
        elif btn == "btn-start":
            self.dismiss({"mode": "background", "session_id": self.session_id})
        elif btn == "btn-view":
            self.dismiss({"mode": "view"})
        elif btn == "btn-stop":
            Docker.stop_background_session(self.project)
            self.notify("Background session stopped")
            self.dismiss(None)
    
    def action_cancel(self):
        self.dismiss(None)


class SessionManagementScreen(ModalScreen):
    """Screen for managing sessions."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("d", "delete_selected", "Delete"),
    ]
    
    def __init__(self, project: Project, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.selected_session = None
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(f"Sessions: {self.project.name}", id="dialog-title")
            yield Static("Select session, then: [Enter] resume, [d] delete", id="session-hint")
            yield DataTable(id="sessions-table", cursor_type="row")
            with Horizontal(id="dialog-buttons"):
                yield Button("New Session", variant="primary", id="btn-new-session")
                yield Button("Resume Selected", id="btn-resume")
                yield Button("Delete Selected", variant="error", id="btn-delete")
                yield Button("Close", id="btn-close")
    
    def on_mount(self):
        self.refresh_sessions()
    
    def refresh_sessions(self):
        table = self.query_one("#sessions-table", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Age", "Files")
        
        sessions = ProjectManager.get_sessions(self.project)
        for s in sessions:
            age = f"{s['age_days']}d" if s['age_days'] > 0 else "today"
            try:
                file_count = len(list(s['path'].glob("*")))
            except Exception:
                file_count = "?"
            table.add_row(s['id'][:12] + "...", age, str(file_count), key=s['id'])
        
        if sessions:
            self.query_one("#session-hint", Static).update(
                f"{len(sessions)} sessions | [Enter] resume, [d] delete"
            )
        else:
            self.query_one("#session-hint", Static).update("No sessions found")
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        """Handle row selection - resume session."""
        if event.row_key:
            self.selected_session = str(event.row_key.value)
            self.dismiss({"action": "resume", "session_id": self.selected_session})
    
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted):
        """Track highlighted row."""
        if event.row_key:
            self.selected_session = str(event.row_key.value)
    
    def on_button_pressed(self, event: Button.Pressed):
        btn = event.button.id
        
        if btn == "btn-close":
            self.dismiss(None)
        elif btn == "btn-new-session":
            self.dismiss({"action": "start_new"})
        elif btn == "btn-resume":
            if self.selected_session:
                self.dismiss({"action": "resume", "session_id": self.selected_session})
            else:
                self.notify("Select a session first", severity="warning")
        elif btn == "btn-delete":
            self.action_delete_selected()
    
    def action_delete_selected(self):
        """Delete selected session."""
        if not self.selected_session:
            self.notify("Select a session first", severity="warning")
            return
        
        session_dir = self.project.openhands_dir / "conversations" / self.selected_session
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
            self.notify(f"Deleted: {self.selected_session[:12]}...")
            self.selected_session = None
            self.refresh_sessions()
        else:
            self.notify("Session not found", severity="error")
    
    def action_cancel(self):
        self.dismiss(None)



# =============================================================================
# MAIN APP
# =============================================================================

class OpenHandsApp(App):
    """Main TUI Application."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-container {
        height: 100%;
    }
    
    #sidebar {
        width: 40;
        border-right: solid $primary;
    }
    
    #sidebar-title {
        text-style: bold;
        padding: 1;
        margin-bottom: 0;
    }
    
    #project-list {
        height: 1fr;
        padding: 0 1;
    }
    
    #actions {
        height: auto;
        padding: 1;
        border-top: solid $primary;
    }
    
    .action-btn {
        width: 100%;
        margin-bottom: 1;
    }
    
    #content {
        padding: 1 2;
    }
    
    #project-header {
        text-style: bold;
        height: auto;
        margin-bottom: 0;
    }
    
    #project-status {
        height: auto;
        margin-bottom: 1;
        color: $text-muted;
    }
    
    #files-label {
        height: auto;
        margin-bottom: 0;
    }
    
    #file-tree {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }
    
    #dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
    }
    
    #dialog-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    #session-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    #session-list {
        height: 20;
        margin-bottom: 1;
        border: solid $primary;
    }
    
    #sessions-table {
        height: 20;
        margin-bottom: 1;
    }
    
    #dialog-buttons {
        margin-top: 1;
        height: auto;
    }
    
    #dialog-buttons Button {
        margin-right: 1;
    }
    
    .settings-row {
        margin-top: 1;
        height: auto;
    }
    
    .settings-row Button {
        margin-right: 1;
        width: 24;
    }
    
    .ralph-actions {
        margin-top: 1;
        height: auto;
    }
    
    .ralph-actions Button {
        margin-right: 2;
    }
    
    .switch-row {
        margin-top: 1;
        height: auto;
    }
    
    .switch-row Switch {
        margin-right: 1;
    }
    
    .section-label {
        text-style: bold;
        margin-top: 1;
    }
    
    TextArea {
        height: 8;
    }
    
    #containers-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    #container-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    #containers-table {
        height: 1fr;
    }
    
    #selected-info {
        margin-top: 1;
        color: $accent;
    }
    
    #container-actions-single {
        margin-top: 1;
    }
    
    #container-actions-single Button {
        margin-right: 1;
    }
    
    #container-actions-all {
        margin-top: 1;
        border-top: solid $primary;
        padding-top: 1;
    }
    
    #container-actions-all Button {
        margin-right: 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("n", "new_project", "New"),
        Binding("s", "start_session", "Start"),
        Binding("r", "ralph", "Ralph"),
        Binding("p", "project_settings", "Settings"),
        Binding("c", "containers", "Containers"),
        Binding("f5", "refresh", "Refresh"),
    ]
    
    selected_project: reactive[Optional[Project]] = reactive(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ProjectManager.init_directories()
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="sidebar"):
                yield Label("Projects", id="sidebar-title")
                yield ListView(id="project-list")
                with Vertical(id="actions"):
                    yield Button("+ New Project", id="btn-new", classes="action-btn")
                    yield Button("> Start Session", id="btn-start", classes="action-btn")
                    yield Button("R Start Ralph", id="btn-ralph", classes="action-btn")
                    yield Button("* Settings", id="btn-settings", classes="action-btn")
                    yield Button("# Containers", id="btn-containers", classes="action-btn")
            with Vertical(id="content"):
                yield Static("Select a project", id="project-header")
                yield Static("", id="project-status")
                yield Label("Workspace Files:", id="files-label")
                yield DirectoryTree(PROJECTS_DIR, id="file-tree")
        yield Footer()
    
    def on_mount(self):
        self.refresh_projects()
        if not Docker.is_available():
            self.notify("Docker not available!", severity="error")
    
    def refresh_projects(self):
        """Refresh project list."""
        list_view = self.query_one("#project-list", ListView)
        list_view.clear()
        
        projects = ProjectManager.list_projects()
        for p in projects:
            item = ListItem(ProjectCard(p))
            item.project_name = p.name
            list_view.append(item)
    
    def on_list_view_selected(self, event: ListView.Selected):
        """Handle project selection."""
        if event.item:
            name = getattr(event.item, 'project_name', None)
            if name:
                self.selected_project = ProjectManager.get_project(name)
                self.update_project_info()
    
    def update_project_info(self):
        """Update project info panel."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        container_status = p.get_container_status()
        llm = p.get_llm_model()
        
        status_display = {
            "running": "[green]running[/]",
            "stopped": "[yellow]stopped[/]",
            "none": "[dim]not created[/]",
            "unknown": "[red]unknown[/]"
        }.get(container_status, container_status)
        
        bg_info = ""
        if container_status == "running" and Docker.is_background_running(p):
            bg_info = " [cyan][BG][/]"
        
        ralph_info = ""
        if p.has_ralph:
            config = p.get_ralph_config()
            r_status = config.get("status", "")
            done, total = p.get_ralph_progress()
            if r_status == "running":
                ralph_info = f"[blue]R[{done}/{total}][/]"
            elif r_status == "complete":
                ralph_info = "[green]OK[/]"
            elif r_status:
                ralph_info = "[yellow]P[/]"
        
        llm_display = llm
        if len(llm) > 25:
            llm_display = llm[:22] + "..."
        
        indicators = " ".join(filter(None, [bg_info, ralph_info]))
        if indicators:
            indicators = f" {indicators}"
        
        self.query_one("#project-header", Static).update(f"[bold]{p.name}[/]{indicators}")
        
        status_text = f"Container: {status_display} | LLM: [cyan]{llm_display}[/]"
        self.query_one("#project-status", Static).update(status_text)
        
        try:
            file_tree = self.query_one("#file-tree", DirectoryTree)
            file_tree.path = p.workspace
            file_tree.reload()
        except Exception:
            pass
    
    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        """Handle file selection."""
        import platform
        
        file_path = str(event.path)
        
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", file_path])
            elif platform.system() == "Windows":
                subprocess.Popen(["start", "", file_path], shell=True)
            else:
                subprocess.Popen(["xdg-open", file_path])
            
            self.notify(f"Opened: {event.path.name}")
        except Exception as e:
            self.notify(f"Failed to open file: {e}", severity="error")
    
    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        btn_id = event.button.id
        
        if btn_id == "btn-new":
            self.action_new_project()
        elif btn_id == "btn-start":
            self.action_start_session()
        elif btn_id == "btn-ralph":
            self.action_ralph()
        elif btn_id == "btn-settings":
            self.action_project_settings()
        elif btn_id == "btn-containers":
            self.action_containers()
    
    def action_new_project(self):
        """Show new project dialog."""
        def handle_result(result):
            if result:
                self.create_project_async(
                    result["name"], 
                    result["template"],
                    result.get("pull", True),
                    result.get("qmd", False)
                )
        
        self.push_screen(NewProjectScreen(), handle_result)
    
    @work(thread=True)
    def create_project_async(self, name: str, template: str, pull: bool, qmd: bool = False):
        """Create project in background thread."""
        if pull:
            self.app.call_from_thread(self.notify, "Pulling latest image... (this may take a while)")
        
        project = ProjectManager.create_project(name, template, pull_image=pull)
        
        if project:
            # Enable QMD if requested
            if qmd:
                qmd_mgr = QMDManager(project)
                success, msg = qmd_mgr.enable()
                if success:
                    self.app.call_from_thread(self.notify, f"Created: {project.name} (QMD enabled)")
                else:
                    self.app.call_from_thread(self.notify, f"Created: {project.name} (QMD failed: {msg})")
            else:
                self.app.call_from_thread(self.notify, f"Created: {project.name}")
            self.app.call_from_thread(self.refresh_projects)
        else:
            self.app.call_from_thread(self.notify, "Failed to create project", severity="error")
    
    def action_start_session(self):
        """Start session."""
        if not self.selected_project:
            self.notify("Select a project first", severity="warning")
            return
        
        def handle_session(result):
            if result:
                session_id = result.get("session")
                self.show_start_mode(session_id)
        
        sessions = ProjectManager.get_sessions(self.selected_project)
        if sessions:
            self.push_screen(SessionSelectScreen(self.selected_project), handle_session)
        else:
            self.show_start_mode(None)
    
    def show_start_mode(self, session_id: Optional[str]):
        """Show start mode selection (background only)."""
        if not self.selected_project:
            return
        
        def handle_mode(result):
            if result:
                mode = result.get("mode")
                if mode == "background":
                    self.start_background_session(result.get("session_id"))
                elif mode == "view":
                    self.view_background_session()
        
        self.push_screen(StartModeScreen(self.selected_project, session_id), handle_mode)
    
    def start_background_session(self, session_id: Optional[str]):
        """Start OpenHands in background and attach to it."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        
        def handle_startup_result(result):
            if result and result.get("success"):
                # Attach to session
                self.exit(result=("view_background", p))
            elif result and result.get("error"):
                self.notify(f"Failed: {result['error']}", severity="error")
        
        config_data = {
            "session_id": session_id,
            "background": True
        }
        
        self.push_screen(
            StartupLogScreen(p, mode="session", config_data=config_data),
            handle_startup_result
        )
    
    def view_background_session(self):
        """Attach to background session."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        
        if not Docker.is_background_running(p):
            self.notify("No background session running", severity="warning")
            return
        
        self.exit(result=("view_background", p))
    
    def _open_ralph_monitor(self, project: Project):
        """Open Ralph monitor screen. Safe to call from thread via call_from_thread."""
        self.push_screen(RalphMonitorScreen(project))
    
    def action_ralph(self):
        """Start, continue, or monitor Ralph."""
        if not self.selected_project:
            self.notify("Select a project first", severity="warning")
            return
        
        p = self.selected_project
        
        # Determine mode
        mode = "new"
        if p.has_ralph:
            config = p.get_ralph_config()
            status = config.get("status", "")
            ralph_actually_running = self._is_ralph_process_running(p)
            
            if status == "running" and ralph_actually_running:
                mode = "running"
            elif status == "running" and not ralph_actually_running:
                # Process died - fix status
                ralph = RalphManager(p)
                ralph.update_config("status", "stopped")
                ralph.add_progress_entry(
                    config.get("currentIteration", 0),
                    "Process was not running (host reboot?), status corrected"
                )
                mode = "stopped"
            elif status == "complete":
                mode = "complete"
            elif status in ["stopped", "paused", "error"]:
                # Check if all tasks are actually complete
                ralph = RalphManager(p)
                done, total = ralph.get_progress()
                if total > 0 and done == total:
                    mode = "complete"
                else:
                    mode = "stopped"
        
        def handle_result(result):
            if not result:
                return
            
            action = result.get("action")
            
            if action == "view_monitor" or action == "view_log":
                self.push_screen(RalphMonitorScreen(p))
            
            elif action == "add_task":
                # Add task to queue (running mode)
                ralph = RalphManager(p)
                task_data = {
                    "description": result["task"],
                    "priority": result.get("priority", "normal")
                }
                ralph.state_manager.add_pending_task(json.dumps(task_data))
                self.notify(f"Task added ({result.get('priority', 'normal')} priority)")
            
            elif action == "resume":
                # Resume without adding task - show startup log
                result["mode"] = "resume"
                self._show_ralph_startup_screen(p, result)
            
            elif action == "add_resume":
                # Add task (optional) and resume
                ralph = RalphManager(p)
                if result.get("task"):
                    task_data = {
                        "description": result["task"],
                        "priority": result.get("priority", "normal")
                    }
                    ralph.state_manager.add_pending_task(json.dumps(task_data))
                result["mode"] = "resume"
                self._show_ralph_startup_screen(p, result)
            
            elif action == "new_mission":
                # New mission - reset plan but keep knowledge
                result["mode"] = "new_mission"
                self._show_ralph_startup_screen(p, result)
            
            elif action == "start":
                # First run - show startup log screen
                self._show_ralph_startup_screen(p, result)
        
        self.push_screen(RalphConfigScreen(p, mode=mode), handle_result)
    
    def _show_ralph_startup_screen(self, p: Project, config_data: dict):
        """Show startup log screen for Ralph."""
        def handle_startup_result(result):
            if result and result.get("success"):
                # Open monitor screen
                self.push_screen(RalphMonitorScreen(p))
            elif result and result.get("error"):
                self.notify(f"Startup failed: {result['error']}", severity="error")
        
        self.push_screen(StartupLogScreen(p, mode="ralph", config_data=config_data), handle_startup_result)
    
    def _is_ralph_process_running(self, project: 'Project') -> bool:
        """Check if Ralph daemon is running inside Docker container."""
        try:
            return Docker.is_ralph_daemon_running(project)
        except Exception:
            return False
    
    def _start_ralph_new_async(self, p: Project, config_data: dict):
        """Start Ralph initialization in background thread."""
        task = config_data.get("task")
        if not task:
            self.notify("Task description required", severity="error")
            return
        
        self.notify("Initializing Ralph... please wait")
        self._do_start_ralph(p, config_data)
    
    @work(thread=True)
    def _do_start_ralph(self, p: Project, config_data: dict):
        """Background worker for Ralph initialization."""
        task = config_data.get("task")
        
        try:
            print(f"[Ralph] Starting for project: {p.name}")
            ralph = RalphManager(p)
            config = RalphConfig(
                task_description=task,
                max_iterations=config_data.get("max_iterations", 0),
                architect_interval=config_data.get("architect_interval", 10)
            )
            
            print(f"[Ralph] Initializing structure...")
            if not ralph.init_structure(config):
                self.app.call_from_thread(self.notify, "Failed to initialize Ralph", severity="error")
                return
            
            print(f"[Ralph] Structure initialized, updating config...")
            # Set status to "starting" while we launch daemon
            ralph.update_config("status", "starting")
            ralph.update_config("condenseInterval", config_data.get("condense_interval", 15))
            
            print(f"[Ralph] Launching daemon process...")
            success = self._launch_ralph_process_sync(p)
            
            if success:
                # Only set running AFTER daemon confirmed started
                ralph.update_config("status", "running")
                self.app.call_from_thread(self.notify, "Ralph started!")
                # Use app method to push screen safely from thread
                self.app.call_from_thread(self._open_ralph_monitor, p)
            else:
                # Mark as stopped since daemon failed
                ralph.update_config("status", "stopped")
                self.app.call_from_thread(self.notify, "Ralph daemon failed to start", severity="error")
            
        except Exception as e:
            print(f"[Ralph] Start failed: {e}")
            print(traceback.format_exc())
            self.app.call_from_thread(self.notify, f"Ralph start failed: {e}", severity="error")
    
    @work(thread=True)
    def _do_resume_ralph(self, p: Project, config_data: dict):
        """Background worker for Ralph resume."""
        try:
            print(f"[Ralph] Resuming for project: {p.name}")
            ralph = RalphManager(p)
            
            # AUTO-FIX: Check for zombie "running" status (STALE)
            current_status = ralph.get_config().get("status", "unknown")
            if current_status == "running":
                process_running = Docker.is_ralph_daemon_running(p)
                if not process_running:
                    print(f"[AUTO-FIX] Status was 'running' but daemon not found - resetting")
                    ralph.update_config("status", "paused")
            
            # Update config - set to "starting" while we launch
            ralph.update_config("status", "starting")
            ralph.update_config("maxIterations", config_data.get("max_iterations", 0))
            ralph.update_config("architectInterval", config_data.get("architect_interval", 10))
            ralph.update_config("condenseInterval", config_data.get("condense_interval", 15))
            ralph.update_config("pauseRequested", False)
            
            # Process any pending tasks
            pending_count = ralph.process_pending_tasks()
            if pending_count > 0:
                self.app.call_from_thread(self.notify, f"Added {pending_count} pending tasks")
            
            prd = ralph.get_prd()
            prd["verified"] = False
            ralph.save_prd(prd)
            
            ralph.add_progress_entry(0, "Resumed - continuing from last state")
            
            print(f"[Ralph] Launching daemon process...")
            success = self._launch_ralph_process_sync(p)
            
            if success:
                ralph.update_config("status", "running")
                self.app.call_from_thread(self.notify, "Ralph resumed!")
                self.app.call_from_thread(self._open_ralph_monitor, p)
            else:
                ralph.update_config("status", "stopped")
                self.app.call_from_thread(self.notify, "Ralph daemon failed to start", severity="error")
            
        except Exception as e:
            print(f"[Ralph] Resume failed: {e}")
            print(traceback.format_exc())
            self.app.call_from_thread(self.notify, f"Ralph resume failed: {e}", severity="error")
    
    @work(thread=True)
    def _do_new_mission_ralph(self, p: Project, config_data: dict):
        """Background worker for new mission."""
        task = config_data.get("task")
        if not task:
            self.app.call_from_thread(self.notify, "Mission description required", severity="error")
            return
        
        try:
            ralph = RalphManager(p)
            
            # Save existing knowledge via Docker
            old_learnings = ralph._read_file("LEARNINGS.md") or ""
            old_architecture = ralph._read_file("ARCHITECTURE.md") or ""
            
            # Extract TASK-XXX references from mission text
            task_refs = re.findall(r'\[?(TASK-\d+)\]?[:\s]+([^\n\[\]]+)', task)
            
            # Create new PRD with planning task
            prd = {
                "projectName": p.name,
                "verified": False,
                "phase": "planning" if not task_refs else "execution",
                "taskDescription": task,
                "userStories": [{
                    "id": "PLAN",
                    "title": "Analyze and create execution plan",
                    "description": "Read the codebase, understand requirements, create detailed plan",
                    "type": "planning",
                    "priority": 0,
                    "passes": bool(task_refs),
                    "dependsOn": []
                }]
            }
            
            # Add explicitly referenced tasks
            if task_refs:
                for i, (task_id, task_title) in enumerate(task_refs):
                    task_title = task_title.strip()
                    if re.match(r'^P\d+:\s*', task_title):
                        task_title = re.sub(r'^P\d+:\s*', '', task_title)
                    prd["userStories"].append({
                        "id": task_id,
                        "title": task_title,
                        "description": f"Task from mission: {task_title}",
                        "type": "feature",
                        "priority": i + 1,
                        "passes": False,
                        "dependsOn": []
                    })
            ralph.save_prd(prd)
            
            # Reset iteration counter - set status to "starting" until daemon confirms
            ralph.update_config("iteration", 0)
            ralph.update_config("currentIteration", 0)
            ralph.update_config("maxIterations", config_data.get("max_iterations", 0))
            ralph.update_config("architectInterval", config_data.get("architect_interval", 10))
            ralph.update_config("condenseInterval", config_data.get("condense_interval", 15))
            ralph.update_config("status", "starting")
            
            # Update mission file via Docker
            timestamp = datetime.now().isoformat()
            mission_content = f"# {task}\n\n*New mission started at {timestamp}*\n\n{task}"
            ralph._write_file("MISSION.md", mission_content)
            
            # Restore knowledge via Docker
            if old_learnings:
                ralph._write_file("LEARNINGS.md", old_learnings)
            if old_architecture:
                ralph._write_file("ARCHITECTURE.md", old_architecture)
            
            # Clear old iteration logs via Docker (backup iterations)
            backup_subdir = timestamp.replace(":", "-")
            Docker.mkdir(ralph.container_name, f"{ralph.CONTAINER_RALPH_DIR}/iterations_backup/{backup_subdir}")
            Docker.exec_in_container(ralph.container_name, f"mv /workspace/.ralph/iterations/* /workspace/.ralph/iterations_backup/{backup_subdir}/ 2>/dev/null || true")
            
            ralph.add_progress_entry(0, f"New mission: {task[:50]}...")
            
            print(f"[Ralph] Launching daemon process...")
            success = self._launch_ralph_process_sync(p)
            
            if success:
                ralph.update_config("status", "running")
                self.app.call_from_thread(self.notify, "New mission started!")
                self.app.call_from_thread(self._open_ralph_monitor, p)
            else:
                ralph.update_config("status", "stopped")
                self.app.call_from_thread(self.notify, "Ralph daemon failed to start", severity="error")
            
        except Exception as e:
            print(f"[Ralph] New mission failed: {e}")
            print(traceback.format_exc())
            self.app.call_from_thread(self.notify, f"New mission failed: {e}", severity="error")
    
    def _launch_ralph_process_sync(self, project: Project) -> bool:
        """Launch Ralph daemon inside Docker container. Returns True on success.
        
        This is a synchronous method called from worker threads.
        """
        print(f"[Ralph] _launch_ralph_process_sync for {project.name}")
        
        # Ensure container is running
        if not Docker.container_running(project.container_name):
            print(f"[Ralph] Container {project.container_name} not running, starting...")
            if not Docker.ensure_container_running(project):
                print(f"[Ralph] Failed to start container!")
                return False
            print(f"[Ralph] Container started")
        
        # Setup Ralph daemon inside container
        print(f"[Ralph] Setting up daemon...")
        if not Docker.setup_ralph_daemon(project):
            print(f"[Ralph] Failed to setup Ralph daemon!")
            return False
        
        # Start Ralph daemon
        print(f"[Ralph] Starting daemon...")
        if Docker.start_ralph_daemon(project):
            print(f"[Ralph] Daemon started successfully!")
            return True
        else:
            print(f"[Ralph] Failed to start Ralph daemon!")
            return False
    
    def _launch_ralph_process(self, project: Project):
        """Legacy wrapper - prefer _launch_ralph_process_sync in workers."""
        return self._launch_ralph_process_sync(project)
    
    def _cleanup_ralph_processes(self, project: Project):
        """Clean up Ralph daemon in container."""
        try:
            if Docker.container_running(project.container_name):
                Docker.stop_ralph_daemon(project)
        except Exception as e:
            # Non-critical - log but continue
            pass
    
    def stop_ralph(self):
        """Stop Ralph daemon in container."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        ralph = RalphManager(p)
        
        # Request pause and save state
        ralph.request_pause()
        
        try:
            if Docker.is_ralph_daemon_running(p):
                Docker.stop_ralph_daemon(p)
                self.notify("Ralph stopped")
            else:
                self.notify("Ralph is not running")
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")
        
        ralph.update_config("status", "stopped")
    
    def action_project_settings(self):
        """Show project settings."""
        if not self.selected_project:
            self.notify("Select a project first", severity="warning")
            return
        
        def handle_result(result):
            if result:
                action = result.get("action")
                if action == "reset":
                    self.reset_project_container()
                elif action == "delete":
                    self.delete_project()
                elif action == "upgrade_openhands":
                    self.upgrade_openhands()
                elif action == "pull_image":
                    self.pull_docker_image()
                elif action == "manage_sessions":
                    self.manage_sessions()
                elif action == "run_mcp_warmup":
                    self.run_mcp_warmup()
                elif action == "restart_gateways":
                    self.restart_mcp_gateways()
                elif action == "upgrade_mcp":
                    self.upgrade_mcp_servers()
                elif action == "llm_changed":
                    self.restart_for_llm_change()
            self.refresh_projects()
            if self.selected_project:
                self.update_project_info()
        
        self.push_screen(ProjectSettingsScreen(self.selected_project), handle_result)
    
    @work(thread=True)
    def pull_docker_image(self):
        """Pull latest Docker image."""
        self.app.call_from_thread(self.notify, f"Pulling: {RUNTIME_IMAGE}... (this may take a while)")
        
        if Docker.pull_image():
            self.app.call_from_thread(self.notify, "Docker image updated! Reset container to use new version.")
        else:
            self.app.call_from_thread(self.notify, "Failed to pull image", severity="error")
    
    @work(thread=True)
    def run_mcp_warmup(self):
        """Run MCP warmup."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        
        self.app.call_from_thread(
            self.notify, 
            "Setting up MCP... This may take 5-10 minutes."
        )
        
        if not Docker.ensure_container_running(p):
            self.app.call_from_thread(self.notify, "Failed to start container", severity="error")
            return
        
        Docker.exec_in_container(
            p.container_name, 
            "rm -f /root/.openhands/.mcp_warmup_done /root/.openhands/.gateway_setup_done", 
            timeout=5
        )
        
        Docker.exec_in_container(
            p.container_name,
            '''
            if [ -f /root/.openhands/.gateway_pids ]; then
                while read pid; do kill $pid 2>/dev/null; done < /root/.openhands/.gateway_pids
                rm -f /root/.openhands/.gateway_pids
            fi
            ''',
            timeout=10
        )
        
        self.app.call_from_thread(self.notify, "Installing MCP tools and dependencies...")
        
        mcp_marker = "/root/.openhands/.mcp_warmup_done"
        warmup_script = f'''
{MCP_WARMUP_SCRIPT}

touch {mcp_marker}
'''
        
        code, output = Docker.exec_in_container(
            p.container_name, 
            warmup_script, 
            timeout=900
        )
        
        if code != 0:
            self.app.call_from_thread(
                self.notify, 
                f"MCP warmup had issues: {output[-200:]}", 
                severity="warning"
            )
        
        self.app.call_from_thread(self.notify, "Starting MCP gateways...")
        
        if Docker.start_mcp_gateways(p):
            running, total = Docker.check_mcp_gateways(p)
            self.app.call_from_thread(
                self.notify, 
                f"MCP setup complete! {running}/{total} gateways running."
            )
        else:
            self.app.call_from_thread(
                self.notify, 
                "MCP gateways failed to start", 
                severity="warning"
            )
    
    @work(thread=True)
    def restart_mcp_gateways(self):
        """Force restart all MCP gateways (kill and restart fresh)."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        
        self.app.call_from_thread(self.notify, "Force restarting all MCP gateways...")
        
        if not Docker.ensure_container_running(p):
            self.app.call_from_thread(self.notify, "Failed to start container", severity="error")
            return
        
        # Use force restart which kills and restarts all gateways
        if Docker.restart_mcp_gateways(p):
            running, total, healthy = Docker.check_mcp_gateways(p)
            self.app.call_from_thread(
                self.notify, 
                f"Gateways restarted! {running}/{total} running, {healthy} healthy."
            )
        else:
            self.app.call_from_thread(
                self.notify, 
                "Failed to restart gateways", 
                severity="error"
            )
    
    @work(thread=True)
    def upgrade_mcp_servers(self):
        """Upgrade MCP servers: install new packages and restart gateways."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        
        self.app.call_from_thread(
            self.notify, 
            "Upgrading MCP servers... This may take 2-5 minutes."
        )
        
        if not Docker.ensure_container_running(p):
            self.app.call_from_thread(self.notify, "Failed to start container", severity="error")
            return
        
        # Read current mcp_servers.json
        mcp_config_path = p.config_dir / "mcp_servers.json"
        if mcp_config_path.exists():
            new_config = mcp_config_path.read_text()
        else:
            new_config = None
        
        # Run upgrade (install packages + restart)
        if Docker.upgrade_mcp_servers(p, new_config):
            running, total, healthy = Docker.check_mcp_gateways(p)
            self.app.call_from_thread(
                self.notify, 
                f"MCP upgrade complete! {healthy}/{total} gateways healthy."
            )
        else:
            self.app.call_from_thread(
                self.notify, 
                "MCP upgrade had issues", 
                severity="warning"
            )
    
    def restart_for_llm_change(self):
        """Restart container after LLM change."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        
        # Stop Ralph daemon if running
        try:
            if Docker.is_ralph_daemon_running(p):
                Docker.stop_ralph_daemon(p)
                self.notify("Stopped Ralph")
                ralph = RalphManager(p)
                ralph.update_config("status", "stopped")
        except Exception:
            pass
        
        if Docker.container_running(p.container_name):
            Docker.stop_container(p.container_name)
            Docker.start_container(p.container_name)
            self.notify("Container restarted with new LLM")
        else:
            self.notify("LLM changed. Will apply on next start.")
    
    def upgrade_openhands(self):
        """Upgrade OpenHands in container."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        
        if not Docker.container_running(p.container_name):
            self.notify("Container not running. Start a session first.", severity="warning")
            return
        
        self.notify("Upgrading OpenHands...")
        
        upgrade_cmd = "uv tool upgrade openhands 2>&1 | tail -5"
        code, output = Docker.exec_in_container(p.container_name, upgrade_cmd, timeout=300)
        
        if code == 0:
            self.notify("OpenHands upgraded!")
        else:
            self.notify(f"Upgrade failed: {output[:100]}", severity="error")
    
    def manage_sessions(self):
        """Show session management."""
        if not self.selected_project:
            return
        
        def handle_result(result):
            if result:
                action = result.get("action")
                if action == "start_new":
                    self.run_session(None)
                elif action == "resume":
                    self.run_session(result.get("session_id"))
                elif action == "delete":
                    session_id = result.get("session_id")
                    if session_id:
                        self.delete_session(session_id)
        
        self.push_screen(SessionManagementScreen(self.selected_project), handle_result)
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        if not self.selected_project:
            return
        
        p = self.selected_project
        session_dir = p.openhands_dir / "conversations" / session_id
        
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
            self.notify(f"Session deleted: {session_id[:12]}...")
        else:
            self.notify("Session not found", severity="error")
    
    def reset_project_container(self):
        """Reset container."""
        if not self.selected_project:
            return
        
        def handle_confirm(confirmed):
            if confirmed and self.selected_project:
                if ProjectManager.reset_container(self.selected_project):
                    self.notify("Container reset. Will reinstall on next start.")
                else:
                    self.notify("Failed to reset container", severity="error")
        
        self.push_screen(
            ConfirmScreen(f"Reset container for '{self.selected_project.name}'?\nThis will reinstall OpenHands and MCP."),
            handle_confirm
        )
    
    def delete_project(self):
        """Delete selected project."""
        if not self.selected_project:
            return
        
        def handle_confirm(confirmed):
            if confirmed and self.selected_project:
                name = self.selected_project.name
                if ProjectManager.delete_project(name):
                    self.notify(f"Deleted: {name}")
                    self.selected_project = None
                    self.refresh_projects()
                    try:
                        self.query_one("#project-header", Static).update("[bold]Welcome[/]")
                        self.query_one("#project-status", Static).update("Select a project from the list")
                        file_tree = self.query_one("#file-tree", DirectoryTree)
                        file_tree.path = PROJECTS_DIR
                        file_tree.reload()
                    except Exception:
                        pass
                else:
                    self.notify("Failed to delete", severity="error")
        
        self.push_screen(
            ConfirmScreen(f"Delete project '{self.selected_project.name}'?\nThis cannot be undone!"),
            handle_confirm
        )
    
    def action_containers(self):
        """Show container management."""
        self.push_screen(ContainersScreen())
    
    def action_refresh(self):
        """Refresh everything."""
        self.refresh_projects()
        if self.selected_project:
            self.update_project_info()
        self.notify("Refreshed")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _run_terminal_command(cmd: List[str], before_msg: str = "", after_msg: str = ""):
    """Run command with full terminal control.
    
    Resets terminal state before running to ensure keyboard shortcuts work
    after exiting Textual TUI.
    """
    import termios
    
    if before_msg:
        print(before_msg)
    print("-" * 50)
    
    # Save and reset terminal state (critical after TUI!)
    old_settings = None
    fd = None
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        # Reset terminal to clean state
        os.system('stty sane 2>/dev/null')
    except Exception:
        pass
    
    try:
        # Build command string for os.system (cleanest terminal passthrough)
        cmd_str = ' '.join(
            f'"{c}"' if ' ' in c or "'" in c else c 
            for c in cmd
        )
        os.system(cmd_str)
    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal
        if old_settings and fd is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
        os.system('stty sane 2>/dev/null')
    
    print("-" * 50)
    if after_msg:
        print(after_msg)
    time.sleep(1)


def view_background_session(project: Project):
    """Attach to background tmux session."""
    cmd = Docker.attach_background_session(project)
    _run_terminal_command(
        cmd,
        before_msg=f">>> Attaching to background session: {project.name}\nTo detach (leave running): Ctrl+B, then D",
        after_msg="--- Detached from background session. Returning to TUI..."
    )


def run_shell(container_name: str):
    """Open shell in container."""
    cmd = ["docker", "exec", "-it", container_name, "bash"]
    _run_terminal_command(
        cmd,
        before_msg=f"> Opening shell in: {container_name}\nType 'exit' to return to TUI",
        after_msg="--- Shell closed. Returning to TUI..."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    # Initialize directories first (projects, templates, etc.)
    ProjectManager.init_directories()
    
    parser = argparse.ArgumentParser(description="OpenHands Manager")
    parser.add_argument("project", nargs="?", help="Project to quick-start")
    parser.add_argument("--help-full", action="store_true", help="Show full help")
    parser.add_argument("--list", action="store_true", help="List projects")
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("--start-ralph", metavar="PROJECT", help="Start Ralph daemon in container")
    args = parser.parse_args()
    
    if args.version:
        print(f"OpenHands Manager v{VERSION}")
        print(f"Max prompt tokens: {MAX_PROMPT_TOKENS}")
        return
    
    if args.start_ralph:
        # Start Ralph daemon in container
        project = ProjectManager.get_project(args.start_ralph)
        if not project:
            print(f"Project not found: {args.start_ralph}")
            sys.exit(1)
        
        print(f"Starting Ralph daemon for: {args.start_ralph}")
        
        # Ensure container is running
        if not Docker.ensure_container_running(project):
            print("Failed to start container")
            sys.exit(1)
        
        # Setup and start daemon
        if not Docker.setup_ralph_daemon(project):
            print("Failed to setup Ralph daemon")
            sys.exit(1)
        
        if Docker.start_ralph_daemon(project):
            print("Ralph daemon started in container!")
            print(f"Monitor with: docker exec -it {project.container_name} tail -f /workspace/.ralph/ralph_daemon.log")
            sys.exit(0)
        else:
            print("Failed to start Ralph daemon")
            sys.exit(1)
    
    if args.list:
        projects = ProjectManager.list_projects()
        if projects:
            print("Projects:")
            for p in projects:
                status = p.get_container_status()
                print(f"  {p.name} ({status})")
        else:
            print("No projects found")
        return
    
    if args.project:
        # Quick start: start background session and attach to it
        project = ProjectManager.get_project(args.project)
        if not project:
            print(f"Project not found: {args.project}")
            sys.exit(1)
        
        print(f"Starting background session for: {project.name}")
        if not Docker.ensure_container_running(project):
            print("[ERROR] Failed to start container")
            sys.exit(1)
        
        if not Docker.is_background_running(project):
            if not Docker.start_background_session(project):
                print("[ERROR] Failed to start background session")
                sys.exit(1)
            print("[OK] Background session started")
            time.sleep(1)
        
        # Attach to the session
        view_background_session(project)
        return
    
    # Run TUI in a loop
    while True:
        app = OpenHandsApp()
        result = app.run()
        
        if result:
            action, *data = result
            if action == "shell":
                container_name = data[0]
                run_shell(container_name)
                continue
            elif action == "view_background":
                project = data[0]
                view_background_session(project)
                continue
        
        break


if __name__ == "__main__":
    main()
