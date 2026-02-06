#!/usr/bin/env python3
"""
Ralph Daemon v3.0 - Enhanced autonomous development daemon.

Major improvements over v1:
- AdaptiveContext: Smart context selection by relevance
- DivergenceDetector: Detect circular patterns early
- Enhanced Condenser: Semantic verification instead of keywords
- LearningsManager with semantic deduplication (unlimited growth)
- HierarchicalMemory with semantic relevance matching
- Optimized for 200K+ token models (conservative prompt budget)

Goal: Every iteration feels like fresh context - no "stupification" over time.

Usage (inside container):
    python3 /workspace/.ralph/ralph_daemon.py
"""

import fcntl  # For file locking
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import shlex
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set
from collections import deque, OrderedDict


# =============================================================================
# AUTO-INSTALL DEPENDENCIES
# =============================================================================

def install_dependencies():
    """Install required packages if not present."""
    required = {
        'sentence_transformers': 'sentence-transformers',
        'numpy': 'numpy',
    }
    missing = []
    
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        print(f"[Ralph] Installing dependencies: {', '.join(missing)}...")
        print("[Ralph] sentence-transformers is ~500MB with torch, please wait...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                *missing, "--break-system-packages"
            ], stderr=subprocess.DEVNULL)
            print("[Ralph] Dependencies installed. Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[Ralph] Failed to install dependencies: {e}")
            print("[Ralph] Please install manually: pip install sentence-transformers")
            sys.exit(1)

install_dependencies()

# =============================================================================
# VERSION
# =============================================================================

DAEMON_VERSION = "3.0.0"  # Git-native state management

# =============================================================================
# GIT STATE INTEGRATION (v5.0)
# =============================================================================

# Try to import git_state module (created by openhands.py v5.0)
GIT_STATE_AVAILABLE = False
_git_state_manager = None
_task_manager = None

def _init_git_state():
    """Initialize git state managers if available."""
    global GIT_STATE_AVAILABLE, _git_state_manager, _task_manager
    
    try:
        # git_state.py is copied to same directory by TUI setup
        import sys
        daemon_dir = str(Path(__file__).parent)
        if daemon_dir not in sys.path:
            sys.path.insert(0, daemon_dir)
        
        from git_state import GitStateManager, TaskManager
        
        workspace = Path("/workspace").resolve()
        _git_state_manager = GitStateManager(workspace)
        # Pass workspace first (git repo root), then prefix, then ralph_dir
        _task_manager = TaskManager(workspace, "oh", RALPH_DIR)
        GIT_STATE_AVAILABLE = True
        print(f"[Ralph] Git-native state enabled (daemon v{DAEMON_VERSION})")
    except Exception as e:
        print(f"[Ralph] Git state not available: {e}")
        GIT_STATE_AVAILABLE = False

# Initialize on module load
try:
    _init_git_state()
except Exception:
    pass

# =============================================================================
# CONSTANTS - Optimized for 250K+ context models
# =============================================================================

# HIGH PRIORITY FIX: Use env var with fallback instead of hardcoded path
# SECURITY FIX: Validate RALPH_DIR is within allowed directory
_ralph_dir_raw = Path(os.environ.get("RALPH_DIR", "/workspace/.ralph")).resolve()
_allowed_base = Path("/workspace").resolve()
try:
    _ralph_dir_raw.relative_to(_allowed_base)
    RALPH_DIR = _ralph_dir_raw
except ValueError:
    print(f"SECURITY ERROR: RALPH_DIR must be within /workspace, got: {_ralph_dir_raw}", file=sys.stderr)
    sys.exit(1)

CONFIG_FILE = RALPH_DIR / "config.json"
PRD_FILE = RALPH_DIR / "prd.json"
MISSION_FILE = RALPH_DIR / "MISSION.md"
LEARNINGS_FILE = RALPH_DIR / "LEARNINGS.md"
ARCHITECTURE_FILE = RALPH_DIR / "ARCHITECTURE.md"
CONDENSED_FILE = RALPH_DIR / "condensed_context.md"
HEARTBEAT_FILE = RALPH_DIR / "heartbeat"
PID_FILE = RALPH_DIR / "ralph_daemon.pid"
LOG_FILE = RALPH_DIR / "ralph_daemon.log"
ITERATIONS_DIR = RALPH_DIR / "iterations"
PROGRESS_FILE = RALPH_DIR / "progress.jsonl"
MEMORY_DIR = RALPH_DIR / "memory"
# REFLECTION_DIR removed - using verification instead

# =============================================================================
# CONTEXT BUDGET - For 250K token models
# =============================================================================
# Total model context: ~250K tokens (~1M chars)
# OpenHands overhead: ~50K tokens
# Our budget: ~100-150K tokens (~400-600K chars)
#
# Allocation:
#   - Mission/Architecture: 60K chars (~15K tokens)
#   - Learnings (condensed): 80K chars (~20K tokens)
#   - Memory context: 60K chars (~15K tokens)  
#   - Current task + history: 40K chars (~10K tokens)
#   - Self-reflection: 20K chars (~5K tokens)
#   - Guardrails: 20K chars (~5K tokens)
#   - Buffer: 40K chars (~10K tokens)
# =============================================================================

CONTEXT_BUDGET_CHARS = 150000  # ~37K tokens - leaves room for model reasoning
MISSION_LIMIT = 10000          # ~2.5K tokens
ARCHITECTURE_LIMIT = 10000     # ~2.5K tokens
LEARNINGS_LIMIT = 30000        # ~7.5K tokens
MEMORY_CONTEXT_LIMIT = 30000   # ~7.5K tokens
TASK_CONTEXT_LIMIT = 20000     # ~5K tokens
# REFLECTION_LIMIT removed
GUARDRAILS_LIMIT = 10000       # ~2.5K tokens

# Semantic thresholds - Tuned for quality
SEMANTIC_DUPLICATE_THRESHOLD = 0.72   # Higher = stricter dedup (was 0.55)
SEMANTIC_COMPACT_THRESHOLD = 0.75     # For cold memory compaction
SEMANTIC_RELEVANCE_THRESHOLD = 0.20   # For selecting relevant content
SEMANTIC_DIVERGENCE_THRESHOLD = 0.85  # High similarity = potential loop
SEMANTIC_CONDENSE_VERIFY = 0.50       # Minimum similarity for fact preservation

# Limits - Prevent unbounded growth
# No learnings limit - semantic search finds relevant from any size
MAX_HOT_MEMORY = 8                    # Recent iterations (full detail)
MAX_WARM_MEMORY = 30                  # Older iterations (summaries)
MAX_COLD_MEMORY = 150                 # Key points (permanent)
MAX_ITERATION_LOGS = 200              # Keep last N iteration logs
MAX_SEMANTIC_CACHE = 8000             # Embedding cache entries

# Maintenance intervals
COMPACT_INTERVAL = 25                 # Compact every N iterations
# REFLECTION_INTERVAL removed - using verification instead
DIVERGENCE_CHECK_WINDOW = 8           # Check last N iterations for loops
KNOWLEDGE_DECAY_DAYS = 7              # Start decaying after N days

# Verification
MAX_VERIFY_RETRIES = 3
MAX_RETRIES = 3
BASE_DELAY = 30
MAX_CONSECUTIVE_ERRORS = 5

# Container timeout (used to cap session timeout for safety)
DOCKER_TIMEOUT = 28800  # 8 hours maximum

# Daemon state
# FIX: Use threading.Event for thread-safe shutdown signaling (instead of raw bool)
import threading
_shutdown_event = threading.Event()
current_process = None
_process_lock = None  # Initialized in main() to avoid import-time threading issues


# =============================================================================
# LOGGING
# =============================================================================

def log(message: str, level: str = "INFO"):
    """Log message with timestamp and level."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_line + "\n")
    except Exception:
        pass


def log_error(message: str):
    log(message, "ERROR")


def log_warning(message: str):
    log(message, "WARN")


def log_debug(message: str):
    log(message, "DEBUG")


# =============================================================================
# ATOMIC FILE OPERATIONS
# =============================================================================

def atomic_write_text(filepath: Path, content: str) -> bool:
    """Write text atomically using tempfile for security.
    
    FIX: Use tempfile.mkstemp for unpredictable temp file names (prevents symlink attacks).
    FIX: Use 0o644 instead of 0o666 for better security.
    FIX: Properly handle FD leak if os.fdopen fails.
    FIX: Check for symlink attacks before writing.
    """
    import tempfile
    filepath = Path(filepath)
    fd = None
    tmp_path = None
    
    # SECURITY FIX: Prevent symlink attacks
    if filepath.is_symlink():
        log_error(f"Security: Refusing to write to symlink: {filepath}")
        return False
    
    # SECURITY FIX: Check parent isn't a symlink
    if filepath.parent.is_symlink():
        log_error(f"Security: Parent directory is symlink: {filepath.parent}")
        return False
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(filepath.parent, 0o755)  # rwxr-xr-x instead of 0o777
        except Exception:
            pass
        
        # Create temp file with random name in same directory
        fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                fd = None  # os.fdopen takes ownership of fd
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            os.rename(tmp_path, filepath)
            tmp_path = None  # Successfully renamed
            
            try:
                os.chmod(filepath, 0o644)  # rw-r--r-- instead of 0o666
            except Exception:
                pass
            return True
        except Exception:
            raise
    except Exception as e:
        # Clean up on error
        if fd is not None:
            try:
                os.close(fd)  # FIX: Close FD if os.fdopen didn't take ownership
            except Exception:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        log_error(f"Write failed for {filepath}: {e}")
        return False


def atomic_write_json(filepath: Path, data: dict) -> bool:
    """Atomically write JSON file."""
    return atomic_write_text(filepath, json.dumps(data, indent=2, ensure_ascii=False))


def safe_read_json(filepath: Path, default: Any = None, max_size: int = 50_000_000) -> Any:
    """Safely read JSON with fallback and size limit.
    
    FIX: Read file first, then check size to avoid TOCTOU race condition
    (file could be swapped between stat and read).
    """
    if not filepath.exists():
        return default
    try:
        # Read file first, then check size (avoids TOCTOU race)
        content = filepath.read_bytes()
        if len(content) > max_size:
            log_error(f"File too large: {filepath} ({len(content)} bytes, max {max_size})")
            return default
        return json.loads(content.decode('utf-8'))
    except Exception:
        return default


# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown.
    
    FIX: Use threading.Event for thread-safe shutdown signaling.
    FIX: Use lock to safely access current_process from signal handler.
    FIX: Don't wait in signal handler - just terminate and let main loop handle cleanup.
    This prevents potential deadlock if main thread is also waiting on the process.
    """
    global current_process, _process_lock
    log(f"Received signal {signum}, shutting down...")
    _shutdown_event.set()  # Thread-safe shutdown signal
    
    # Safely get and clear process reference with lock
    proc = None
    if _process_lock:
        with _process_lock:
            proc = current_process
            current_process = None  # Mark as "being handled"
    else:
        proc = current_process
        current_process = None
    
    if proc:
        try:
            # Non-blocking terminate - main loop will handle cleanup
            if proc.poll() is None:
                log("Terminating current iteration...")
                proc.terminate()
                # Don't wait here - main loop will detect _shutdown_event
        except Exception:
            pass


def write_heartbeat():
    """Write heartbeat timestamp."""
    try:
        HEARTBEAT_FILE.write_text(str(int(time.time())))
        os.chmod(HEARTBEAT_FILE, 0o644)  # SECURITY: was 0o666
    except Exception:
        pass


def add_progress(iteration: int, message: str):
    """Add progress entry."""
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "message": message
        }
        with open(PROGRESS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def read_config() -> dict:
    """Read config with validation to prevent infinite loops on corruption.
    
    FIX: Validate config fields to prevent daemon spin on corrupted data.
    """
    default = {"status": "paused", "currentIteration": 0}
    data = safe_read_json(CONFIG_FILE, default)
    
    # Validate required fields
    if not isinstance(data.get("currentIteration"), int):
        data["currentIteration"] = 0
    # FIX: Explicit None check for clarity (None not in list is True, but explicit is better)
    status = data.get("status")
    valid_statuses = ["running", "paused", "stopped", "complete", "error", "initialized", "starting"]
    if status is None or status not in valid_statuses:
        log_warning(f"Invalid status '{status}', resetting to 'paused'")
        data["status"] = "paused"
    
    return data


def save_config(config: dict):
    if not atomic_write_json(CONFIG_FILE, config):
        log_error("Failed to save config")


def update_config(key: str, value):
    """Update config with file locking to prevent race conditions.
    
    FIX: Uses fcntl.flock for atomic read-modify-write, same pattern as mark_task_done().
    This prevents concurrent updates (e.g., TUI + daemon) from losing changes.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Ensure file exists
            if not CONFIG_FILE.exists():
                config = {"status": "paused", "currentIteration": 0}
                config[key] = value
                save_config(config)
                return
            
            with open(CONFIG_FILE, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    content = f.read()
                    config = json.loads(content) if content.strip() else {"status": "paused", "currentIteration": 0}
                    config[key] = value
                    
                    f.seek(0)
                    f.truncate()
                    json.dump(config, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                return
        except (IOError, OSError, json.JSONDecodeError) as e:
            log_warning(f"Config update failed, retrying ({attempt+1}/{max_retries}): {e}")
            time.sleep(0.1 * (attempt + 1))
    
    log_error(f"Failed to update config key '{key}' after {max_retries} retries")


def read_prd() -> dict:
    return safe_read_json(PRD_FILE, {"userStories": [], "verified": False})


def save_prd(prd: dict, commit: bool = False, message: str = None):
    """Save PRD with optional git commit."""
    if not atomic_write_json(PRD_FILE, prd):
        log_error("Failed to save PRD")
        return
    
    # Commit to git if git state available
    if commit and GIT_STATE_AVAILABLE and _git_state_manager:
        try:
            _git_state_manager._run_git("add", str(PRD_FILE))
            if message:
                _git_state_manager._run_git("commit", "-m", message, "--allow-empty")
        except Exception as e:
            log_warning(f"Git commit failed: {e}")


# =============================================================================
# GIT-NATIVE STATE FUNCTIONS (v3.0)
# =============================================================================

def get_git_iteration() -> int:
    """Get current iteration from git state."""
    if GIT_STATE_AVAILABLE and _git_state_manager:
        return _git_state_manager.get_iteration()
    # Fallback: read from config
    config = read_config()
    return config.get("current_iteration", 0)


def set_git_iteration(n: int):
    """Set current iteration in git state."""
    if GIT_STATE_AVAILABLE and _git_state_manager:
        _git_state_manager.set_iteration(n)


def get_git_task() -> str:
    """Get current task ID from git state."""
    if GIT_STATE_AVAILABLE and _git_state_manager:
        return _git_state_manager.get_current_task()
    return ""


def _sanitize_git_text(text: str, max_length: int = 200) -> str:
    """Sanitize text for use in git commit/tag messages.
    
    Removes null bytes and control characters, limits length.
    """
    if not text:
        return ""
    # Remove null bytes and control characters (except newline for multi-line)
    text = ''.join(c for c in text if c == '\n' or (ord(c) >= 32 and ord(c) != 127))
    return text[:max_length]


def set_git_task(task_id: str):
    """Set current task ID in git state."""
    if GIT_STATE_AVAILABLE and _git_state_manager:
        # Sanitize task_id
        safe_task_id = _sanitize_git_text(str(task_id), 50)
        _git_state_manager.set_current_task(safe_task_id)


def commit_iteration_to_git(iteration: int, task_id: str, summary: str) -> bool:
    """Commit iteration progress to git with structured message."""
    if not GIT_STATE_AVAILABLE or not _git_state_manager:
        return False
    
    # Validate and sanitize inputs
    if not isinstance(iteration, int) or iteration < 0:
        log_warning(f"Invalid iteration number: {iteration}")
        return False
    
    safe_task_id = _sanitize_git_text(str(task_id), 50)
    safe_summary = _sanitize_git_text(str(summary), 200)
    
    try:
        return _git_state_manager.commit_iteration(iteration, safe_task_id, safe_summary)
    except Exception as e:
        log_warning(f"Git commit failed: {e}")
        return False


def add_learning_to_git(learning: str) -> bool:
    """Add learning as git note."""
    if not GIT_STATE_AVAILABLE or not _git_state_manager:
        return False
    
    safe_learning = _sanitize_git_text(str(learning), 500)
    if not safe_learning:
        return False
    
    try:
        return _git_state_manager.add_learning(safe_learning)
    except Exception as e:
        log_warning(f"Git learning add failed: {e}")
        return False


def get_learnings_from_git(limit: int = 100) -> List[str]:
    """Get learnings from git notes."""
    if GIT_STATE_AVAILABLE and _git_state_manager:
        return _git_state_manager.get_learnings(limit=limit)
    return []


def save_checkpoint_to_git(iteration: int, task_id: str, status: str = "in_progress") -> bool:
    """Save checkpoint as git tag."""
    if GIT_STATE_AVAILABLE and _git_state_manager:
        return _git_state_manager.save_checkpoint(iteration, task_id, status)
    return False


def load_checkpoint_from_git():
    """Load latest checkpoint from git tag."""
    if GIT_STATE_AVAILABLE and _git_state_manager:
        return _git_state_manager.load_checkpoint()
    return None


def get_task_by_id(task_id: str):
    """Get task by ID from TaskManager."""
    if GIT_STATE_AVAILABLE and _task_manager:
        return _task_manager.get_task(task_id)
    return None


def get_next_pending_task():
    """Get next pending task from TaskManager."""
    if GIT_STATE_AVAILABLE and _task_manager:
        return _task_manager.get_next_task()
    return None


def set_task_status_by_id(task_id: str, status: str) -> bool:
    """Set task status in TaskManager."""
    if GIT_STATE_AVAILABLE and _task_manager:
        return _task_manager.set_task_status(task_id, status)
    return False


def get_tasks_for_prompt() -> str:
    """Get tasks formatted for LLM prompt."""
    if GIT_STATE_AVAILABLE and _task_manager:
        return _task_manager.to_prompt_format()
    # Fallback: format old PRD
    prd = read_prd()
    lines = ["## Tasks\n"]
    for story in prd.get("userStories", []):
        status = "✅" if story.get("passes") else "⏳"
        lines.append(f"### {status} {story.get('id', '?')}: {story.get('title', '')}")
        if story.get("description"):
            lines.append(story["description"])
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# SEMANTIC SEARCH - Enhanced with LRU cache
# =============================================================================

class SemanticSearch:
    """Semantic search with LRU cache and batch operations.
    
    FIX: Uses OrderedDict for O(1) LRU operations instead of deque which had O(n) remove().
    """
    
    _model = None
    _model_loaded = False
    
    def __init__(self):
        self.cache_file = RALPH_DIR / "semantic_cache.json"
        # FIX: Use OrderedDict for O(1) move_to_end() instead of deque.remove() which is O(n)
        self._embeddings_cache: OrderedDict = OrderedDict()
        self._load_cache()
    
    @classmethod
    def _get_model(cls):
        """Lazy-load Sentence Transformer model."""
        if cls._model_loaded:
            return cls._model
        
        cls._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer('all-mpnet-base-v2')
            log("Loaded sentence-transformers: all-mpnet-base-v2")
        except Exception as e:
            log_error(f"Failed to load model: {e}")
            raise
        return cls._model
    
    def _load_cache(self):
        """Load cache from disk into OrderedDict (preserves LRU order)."""
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text())
                # Load only recent entries, preserving order
                keys = list(data.keys())[-MAX_SEMANTIC_CACHE:]
                self._embeddings_cache = OrderedDict((k, data[k]) for k in keys)
            except Exception:
                self._embeddings_cache = OrderedDict()
    
    def _save_cache(self):
        """Save cache to disk (OrderedDict preserves LRU order naturally)."""
        try:
            # OrderedDict preserves insertion order, just save directly
            self.cache_file.write_text(json.dumps(dict(self._embeddings_cache)))
            os.chmod(self.cache_file, 0o644)
        except Exception:
            pass
    
    def _text_hash(self, text: str) -> str:
        # FIX: Use full SHA256 instead of truncated MD5 to avoid cache key collisions
        # (MD5[:16] = 64 bits has non-negligible collision probability with 8000 entries)
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _get_embedding(self, text: str):
        """Get embedding with O(1) LRU cache using OrderedDict.
        
        FIX: Uses OrderedDict.move_to_end() which is O(1) instead of
        deque.remove() which was O(n). With 8000 cache entries this is significant.
        """
        import numpy as np
        model = self._get_model()
        
        text_hash = self._text_hash(text)
        
        if text_hash in self._embeddings_cache:
            # O(1) move to end (most recently used)
            self._embeddings_cache.move_to_end(text_hash)
            return np.array(self._embeddings_cache[text_hash], dtype=np.float32)
        
        # Evict oldest entries if at limit (O(1) per eviction)
        while len(self._embeddings_cache) >= MAX_SEMANTIC_CACHE:
            self._embeddings_cache.popitem(last=False)  # Remove oldest
        
        # Compute new embedding
        embedding = model.encode(text, convert_to_numpy=True)
        self._embeddings_cache[text_hash] = embedding.tolist()
        
        return embedding.astype(np.float32)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity."""
        from sentence_transformers.util import cos_sim
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        return float(cos_sim(emb1, emb2)[0][0])
    
    def find_similar(self, query: str, documents: List[str], top_k: int = 5,
                     threshold: float = 0.1) -> List[Tuple[int, float, str]]:
        """Find most similar documents."""
        if not documents:
            return []
        
        from sentence_transformers.util import cos_sim
        import numpy as np
        
        query_emb = self._get_embedding(query)
        doc_embs = np.array([self._get_embedding(d) for d in documents])
        
        similarities = cos_sim(query_emb, doc_embs)[0].numpy()
        
        results = [(i, float(s), d) for i, (s, d) in enumerate(zip(similarities, documents)) if s >= threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def is_duplicate(self, new_text: str, existing_texts: List[str], threshold: float = SEMANTIC_DUPLICATE_THRESHOLD) -> bool:
        """Check if semantically duplicate."""
        if not existing_texts:
            return False
        similar = self.find_similar(new_text, existing_texts, top_k=1, threshold=threshold)
        return len(similar) > 0
    
    def batch_similarities(self, query: str, documents: List[str]) -> List[float]:
        """Get similarities for all documents."""
        if not documents:
            return []
        
        from sentence_transformers.util import cos_sim
        import numpy as np
        
        query_emb = self._get_embedding(query)
        doc_embs = np.array([self._get_embedding(d) for d in documents])
        
        return cos_sim(query_emb, doc_embs)[0].numpy().tolist()
    
    def flush_cache(self):
        """Save cache to disk."""
        self._save_cache()


# =============================================================================
# HIERARCHICAL MEMORY - Optimized for large context
# =============================================================================

class HierarchicalMemory:
    """Three-tier memory with relevance-based retrieval."""
    
    def __init__(self, semantic: SemanticSearch):
        MEMORY_DIR.mkdir(exist_ok=True)
        self.semantic = semantic
        self.hot_file = MEMORY_DIR / "hot.json"
        self.warm_file = MEMORY_DIR / "warm.json"
        self.cold_file = MEMORY_DIR / "cold.json"
    
    def _load(self, filepath: Path) -> List:
        return safe_read_json(filepath, [])
    
    def _save(self, filepath: Path, data: List):
        try:
            filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            os.chmod(filepath, 0o644)  # SECURITY: was 0o666
        except Exception as e:
            log_error(f"Failed to save {filepath}: {e}")
    
    def add_iteration(self, iteration: int, task_id: str, summary: str, 
                      details: str, key_points: List[str] = None, outcome: str = ""):
        """Add iteration to memory with outcome tracking."""
        entry = {
            'iteration': iteration,
            'task_id': task_id,
            'summary': summary,
            'details': details[:3000],  # Limit details size
            'key_points': (key_points or [])[:10],
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        }
        
        hot = self._load(self.hot_file)
        hot.append(entry)
        
        # Move old to warm
        while len(hot) > MAX_HOT_MEMORY:
            old = hot.pop(0)
            self._add_to_warm(old)
        
        self._save(self.hot_file, hot)
    
    def _add_to_warm(self, entry: dict):
        """Move to warm tier with compression.
        
        FIX: Batch cold promotions to avoid O(n) file I/O operations.
        """
        warm = self._load(self.warm_file)
        warm_entry = {
            'iteration': entry['iteration'],
            'task_id': entry['task_id'],
            'summary': entry['summary'][:300],
            'key_points': entry.get('key_points', [])[:5],
            'outcome': entry.get('outcome', ''),
            'timestamp': entry['timestamp']
        }
        warm.append(warm_entry)
        
        # Collect all entries to promote to cold in one batch
        cold_promotions = []
        while len(warm) > MAX_WARM_MEMORY:
            cold_promotions.append(warm.pop(0))
        
        # Batch update cold storage (single file I/O instead of O(n))
        if cold_promotions:
            self._batch_add_to_cold(cold_promotions)
        
        self._save(self.warm_file, warm)
    
    def _add_to_cold(self, entry: dict):
        """Extract key points to permanent storage."""
        cold = self._load(self.cold_file)
        
        for point in entry.get('key_points', []):
            if point and point not in cold:
                cold.append(point)
        
        # Add outcome as key point if significant
        outcome = entry.get('outcome', '')
        if outcome and outcome not in ['working', 'unknown'] and outcome not in cold:
            cold.append(f"[{entry['task_id']}] {outcome}")
        
        # Deduplicate cold storage
        cold = list(dict.fromkeys(cold))[-MAX_COLD_MEMORY:]
        self._save(self.cold_file, cold)
    
    def _batch_add_to_cold(self, entries: List[dict]):
        """Batch add multiple entries to cold storage (single file I/O).
        
        FIX: More efficient than calling _add_to_cold() for each entry.
        """
        if not entries:
            return
        
        cold = self._load(self.cold_file)
        
        for entry in entries:
            for point in entry.get('key_points', []):
                if point and point not in cold:
                    cold.append(point)
            
            # Add outcome as key point if significant
            outcome = entry.get('outcome', '')
            if outcome and outcome not in ['working', 'unknown'] and outcome not in cold:
                cold.append(f"[{entry['task_id']}] {outcome}")
        
        # Deduplicate cold storage
        cold = list(dict.fromkeys(cold))[-MAX_COLD_MEMORY:]
        self._save(self.cold_file, cold)
    
    def get_context(self, current_task: str, max_chars: int = MEMORY_CONTEXT_LIMIT) -> str:
        """Get contextually relevant memory."""
        parts = []
        remaining = max_chars
        
        # Hot tier - always include recent
        hot = self._load(self.hot_file)
        if hot:
            hot_text = "## Recent Iterations\n"
            for entry in hot[-5:]:  # Last 5 with details
                hot_text += f"\n### Iteration {entry['iteration']}: {entry['task_id']}\n"
                hot_text += f"Outcome: {entry.get('outcome', 'unknown')}\n"
                hot_text += entry.get('summary', '')[:500]
                if entry.get('key_points'):
                    hot_text += "\nKey points:\n" + '\n'.join(f"- {p}" for p in entry['key_points'][:3])
                hot_text += "\n"
            
            if len(hot_text) <= remaining:
                parts.append(hot_text)
                remaining -= len(hot_text)
        
        # Warm tier - select by relevance
        if remaining > 5000 and current_task:
            warm = self._load(self.warm_file)
            if warm:
                # Score by relevance to current task
                warm_texts = [f"{e['task_id']} {e['summary']}" for e in warm]
                similarities = self.semantic.batch_similarities(current_task, warm_texts)
                
                scored = list(zip(warm, similarities))
                scored.sort(key=lambda x: x[1], reverse=True)
                
                relevant = [e for e, s in scored[:5] if s > SEMANTIC_RELEVANCE_THRESHOLD]
                
                if relevant:
                    warm_text = "\n## Related Past Work\n"
                    for entry in relevant:
                        warm_text += f"- [{entry['task_id']}] {entry['summary'][:150]} ({entry.get('outcome', '?')})\n"
                    
                    if len(warm_text) <= remaining:
                        parts.append(warm_text)
                        remaining -= len(warm_text)
        
        # Cold tier - key points by relevance
        if remaining > 2000 and current_task:
            cold = self._load(self.cold_file)
            if cold:
                # Filter by relevance
                similarities = self.semantic.batch_similarities(current_task, cold)
                relevant_points = [p for p, s in zip(cold, similarities) if s > 0.15][-20:]
                
                if relevant_points:
                    cold_text = "\n## Key Learnings\n"
                    cold_text += '\n'.join(f"- {p}" for p in relevant_points)
                    
                    if len(cold_text) <= remaining:
                        parts.append(cold_text)
        
        return '\n'.join(parts)
    
    def add_permanent(self, point: str):
        """Add directly to cold storage."""
        cold = self._load(self.cold_file)
        point = point.strip()
        if point and point not in cold:
            cold.append(point)
            cold = cold[-MAX_COLD_MEMORY:]
            self._save(self.cold_file, cold)
    
    def compact(self, semantic: SemanticSearch):
        """Compact all tiers, removing near-duplicates."""
        # Compact cold storage
        cold = self._load(self.cold_file)
        if len(cold) < 30:
            return
        
        unique = []
        for point in cold:
            if not unique or not semantic.is_duplicate(point, unique, threshold=SEMANTIC_COMPACT_THRESHOLD):
                unique.append(point)
        
        if len(unique) < len(cold):
            log(f"Memory compaction: {len(cold)} -> {len(unique)} cold points")
            self._save(self.cold_file, unique)
        
        # Compact warm storage summaries
        warm = self._load(self.warm_file)
        if len(warm) > MAX_WARM_MEMORY:
            self._save(self.warm_file, warm[-MAX_WARM_MEMORY:])
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "hot": len(self._load(self.hot_file)),
            "warm": len(self._load(self.warm_file)),
            "cold": len(self._load(self.cold_file))
        }


# =============================================================================
# LEARNINGS MANAGER - With hard limits and smart compaction
# =============================================================================

class LearningsManager:
    """Manage learnings with strict limits and semantic deduplication."""
    
    def __init__(self, semantic: SemanticSearch):
        self.semantic = semantic
        self._entries: List[Dict] = []
        self._load()
    
    def _load(self):
        """Load learnings with metadata."""
        if not LEARNINGS_FILE.exists():
            return
        
        content = LEARNINGS_FILE.read_text()
        sections = re.split(r'\n(?=##\s)', content)
        
        self._entries = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Try to extract iteration number
            match = re.search(r'Iteration (\d+)', section)
            iteration = int(match.group(1)) if match else 0
            
            self._entries.append({
                'content': section,
                'iteration': iteration,
                'timestamp': datetime.now().isoformat()
            })
    
    def _save(self):
        """Save learnings to file with hard limit to prevent OOM."""
        # HIGH PRIORITY FIX: Hard limit of ~50MB to prevent OOM
        MAX_LEARNINGS_SIZE_MB = 50
        MAX_LEARNINGS_SIZE_BYTES = MAX_LEARNINGS_SIZE_MB * 1024 * 1024
        
        content = '\n\n'.join(e['content'] for e in self._entries)
        
        # Archive if exceeds limit
        content_bytes = content.encode('utf-8')
        if len(content_bytes) > MAX_LEARNINGS_SIZE_BYTES:
            log_warning(f"Learnings file exceeds {MAX_LEARNINGS_SIZE_MB}MB, archiving old entries")
            # Keep only last 50% of entries
            keep_count = len(self._entries) // 2
            self._entries = self._entries[-keep_count:]
            content = '\n\n'.join(e['content'] for e in self._entries)
            
            # Archive old content
            archive_path = LEARNINGS_FILE.parent / f"LEARNINGS_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            try:
                archive_path.write_text(content)
                log(f"Archived learnings to {archive_path}")
            except Exception as e:
                log_error(f"Failed to archive learnings: {e}")
        
        try:
            LEARNINGS_FILE.write_text(content)
            os.chmod(LEARNINGS_FILE, 0o644)  # SECURITY: was 0o666
        except Exception as e:
            log_error(f"Failed to save learnings: {e}")
    
    def add(self, learning: str, iteration: int = 0) -> bool:
        """Add learning if not duplicate. Returns True if added."""
        learning = learning.strip()
        if not learning:
            return False
        
        # Check for semantic duplicates
        existing_texts = [e['content'] for e in self._entries]
        if existing_texts and self.semantic.is_duplicate(learning, existing_texts, threshold=SEMANTIC_DUPLICATE_THRESHOLD):
            log_debug(f"Skipped duplicate learning")
            return False
        
        # Format with metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        formatted = f"## Iteration {iteration} ({timestamp})\n{learning}"
        
        self._entries.append({
            'content': formatted,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat()
        })
        
        # HIGH PRIORITY FIX: Hard limit on entries to prevent OOM
        MAX_ENTRIES = 10000
        if len(self._entries) > MAX_ENTRIES:
            # Remove oldest 10% when limit exceeded
            remove_count = len(self._entries) // 10
            self._entries = self._entries[remove_count:]
            log_warning(f"Learnings entries exceeded {MAX_ENTRIES}, removed {remove_count} oldest")
        
        self._save()
        return True
    
    def get_relevant(self, query: str, max_chars: int = LEARNINGS_LIMIT) -> str:
        """Get relevant learnings for query."""
        if not self._entries:
            return ""
        
        texts = [e['content'] for e in self._entries]
        similarities = self.semantic.batch_similarities(query, texts)
        
        # Sort by relevance
        scored = list(zip(self._entries, similarities))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Build result within budget
        result = []
        chars = 0
        
        for entry, score in scored:
            if score < SEMANTIC_RELEVANCE_THRESHOLD:
                continue
            content = entry['content']
            if chars + len(content) > max_chars:
                break
            result.append(content)
            chars += len(content)
        
        return '\n\n'.join(result)
    
    def get_all(self, max_chars: int = LEARNINGS_LIMIT) -> str:
        """Get all learnings within limit."""
        result = []
        chars = 0
        
        for entry in reversed(self._entries):  # Newest first
            content = entry['content']
            if chars + len(content) > max_chars:
                break
            result.append(content)
            chars += len(content)
        
        return '\n\n'.join(reversed(result))
    
    def count(self) -> int:
        return len(self._entries)


# =============================================================================
# CONTEXT CONDENSER - Enhanced with semantic verification
# =============================================================================

class ContextCondenser:
    """Condense context with semantic verification."""
    
    def __init__(self, semantic: SemanticSearch):
        self.semantic = semantic
        self.history_file = RALPH_DIR / "condense_history.json"
        self.last_iteration = 0
        self._load_state()
    
    def _load_state(self):
        history = safe_read_json(self.history_file, {})
        self.last_iteration = history.get('last_iteration', 0)
    
    def _save_state(self, iteration: int):
        atomic_write_json(self.history_file, {
            'last_iteration': iteration,
            'timestamp': datetime.now().isoformat()
        })
        self.last_iteration = iteration
    
    def should_condense(self, iteration: int, config: dict, 
                        context_size: int = 0, next_is_architect: bool = False) -> Tuple[bool, str]:
        """Check if condensation needed."""
        interval = config.get('condenseInterval', COMPACT_INTERVAL)
        if interval <= 0:
            return False, "disabled"
        
        since_last = iteration - self.last_iteration
        
        if since_last >= interval:
            return True, f"interval ({since_last} iters)"
        
        if config.get('condenseBeforeArchitect', True) and next_is_architect and since_last >= 5:
            return True, "before architect"
        
        # Context size trigger - using our budget
        if context_size > CONTEXT_BUDGET_CHARS * 0.8:
            return True, f"context size ({context_size} chars)"
        
        return False, ""
    
    def extract_critical_facts(self, content: str) -> List[Dict]:
        """Extract facts that must be preserved."""
        # FIX: Limit input size to prevent ReDoS (regex denial of service)
        MAX_CONTENT_SIZE = 100000  # 100KB max
        if len(content) > MAX_CONTENT_SIZE:
            content = content[:MAX_CONTENT_SIZE]
        
        facts = []
        
        # Errors with context
        for match in re.finditer(r'(?:error|failed|exception|bug|fix|crash)[:\s]+([^\n]{20,200})', content, re.I):
            facts.append({'type': 'ERROR', 'content': match.group(1).strip()})
        
        # Decisions
        for match in re.finditer(r'(?:decided|chose|because|will use|using)[:\s]+([^\n]{20,200})', content, re.I):
            facts.append({'type': 'DECISION', 'content': match.group(1).strip()})
        
        # Completions
        for match in re.finditer(r'(?:completed|finished|done|implemented)[:\s]+([^\n]{10,150})', content, re.I):
            facts.append({'type': 'COMPLETED', 'content': match.group(1).strip()})
        
        # Architecture changes
        for match in re.finditer(r'(?:created|added|modified|changed|refactored)[:\s]+([^\n]{10,150})', content, re.I):
            facts.append({'type': 'CHANGE', 'content': match.group(1).strip()})
        
        # Deduplicate
        seen = set()
        unique = []
        for fact in facts:
            key = fact['content'].lower()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(fact)
        
        return unique[:30]  # Limit to 30 facts
    
    def verify_condensation(self, original: str, condensed: str) -> Tuple[bool, List[Dict]]:
        """Verify critical facts preserved using SEMANTIC similarity."""
        facts = self.extract_critical_facts(original)
        if not facts:
            return True, []
        
        missing = []
        preserved = 0
        
        for fact in facts:
            fact_text = f"{fact['type']}: {fact['content']}"
            
            # Use semantic similarity instead of keyword matching
            similarity = self.semantic.compute_similarity(fact_text, condensed)
            
            # FIX: Add epsilon for float comparison tolerance
            if similarity >= SEMANTIC_CONDENSE_VERIFY - 1e-9:
                preserved += 1
            else:
                missing.append(fact)
        
        # Require 80% preservation (was 70%)
        is_valid = (preserved / len(facts)) >= 0.80 if facts else True
        
        log(f"Condense verification: {preserved}/{len(facts)} facts preserved")
        return is_valid, missing
    
    def save_condensed(self, iteration: int, condensed: str, original: str = ""):
        """Save condensed context with auto-recovery of missing facts."""
        
        if original:
            is_valid, missing = self.verify_condensation(original, condensed)
            
            if missing:
                # Auto-append missing critical facts
                condensed += "\n\n## Auto-Preserved Critical Facts\n"
                for fact in missing[:15]:
                    condensed += f"- [{fact['type']}] {fact['content']}\n"
                log(f"Auto-preserved {len(missing)} missing facts")
        
        # Backup old
        if CONDENSED_FILE.exists():
            backup_dir = RALPH_DIR / "condense_backups"
            backup_dir.mkdir(exist_ok=True)
            backup = backup_dir / f"condensed_{self.last_iteration}.md"
            try:
                shutil.copy2(CONDENSED_FILE, backup)
                # Keep last 3 backups only
                for old in sorted(backup_dir.glob("condensed_*.md"))[:-3]:
                    old.unlink()
            except Exception:
                pass
        
        # Save new
        header = f"# Condensed Context (Iteration {iteration})\n"
        header += f"# Generated: {datetime.now().isoformat()}\n\n"
        
        try:
            CONDENSED_FILE.write_text(header + condensed)
            os.chmod(CONDENSED_FILE, 0o644)  # SECURITY: was 0o666
            self._save_state(iteration)
            log(f"Saved condensed: {len(condensed)} chars")
        except Exception as e:
            log_error(f"Failed to save condensed: {e}")
    
    def get_condensed(self) -> str:
        if CONDENSED_FILE.exists():
            return CONDENSED_FILE.read_text()
        return ""
    
    def parse_condensed(self, output: str) -> Optional[str]:
        """Extract condensed content from output."""
        match = re.search(r'```condensed\s*\n(.*?)\n```', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        match = re.search(r'(## Mission Summary.*)', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        if '## ' in output:
            idx = output.index('## ')
            return output[idx:].strip()
        
        return None


# =============================================================================
# DIVERGENCE DETECTOR - NEW: Detect circular patterns
# =============================================================================

class DivergenceDetector:
    """
    Detect when agent is stuck in circular patterns.
    
    Signals:
    - High similarity between recent iterations
    - Same errors repeating
    - No task progress despite activity
    """
    
    def __init__(self, semantic: SemanticSearch):
        self.semantic = semantic
        self.history_file = RALPH_DIR / "divergence_history.json"
        self.recent_outputs: deque = deque(maxlen=DIVERGENCE_CHECK_WINDOW)
    
    def record_iteration(self, iteration: int, output_summary: str, outcome: str) -> int:
        """Record iteration for pattern detection. Returns iteration number for later updates.
        
        FIX: Return iteration number instead of dict reference to avoid invalidation
        when deque evicts entries (maxlen exceeded).
        """
        entry = {
            'iteration': iteration,
            'summary': output_summary[:1000],
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        }
        self.recent_outputs.append(entry)
        return iteration  # Return iteration number, not reference
    
    def check_divergence(self) -> Tuple[bool, str, str]:
        """
        Check for circular patterns.
        Returns: (is_diverging, reason, suggested_action)
        """
        if len(self.recent_outputs) < 5:
            return False, "", ""
        
        outputs = list(self.recent_outputs)
        
        # Check 1: High similarity between consecutive iterations
        similarities = []
        for i in range(len(outputs) - 1):
            sim = self.semantic.compute_similarity(
                outputs[i]['summary'],
                outputs[i + 1]['summary']
            )
            similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities)
        
        if avg_similarity > SEMANTIC_DIVERGENCE_THRESHOLD:
            # High similarity = potentially going in circles
            # Check if there's actual progress
            outcomes = [o['outcome'] for o in outputs]
            progress_outcomes = ['task_done', 'verified_pass', 'planning_done']
            has_progress = any(o in progress_outcomes for o in outcomes)
            
            if not has_progress:
                return True, f"circular_pattern (avg_sim={avg_similarity:.2f})", "force_new_approach"
        
        # Check 2: Same outcome repeating
        # FIX: Guard against empty list and falsy outcomes
        outcomes = [o['outcome'] for o in outputs[-5:]]
        if outcomes and outcomes[0]:  # Ensure list is non-empty and first element is truthy
            if outcomes.count(outcomes[0]) >= 4 and outcomes[0] not in ['task_done', 'verified_pass']:
                return True, f"repeated_outcome ({outcomes[0]} x{outcomes.count(outcomes[0])})", "escalate_to_architect"
        
        # Check 3: Error patterns
        error_outputs = [o for o in outputs if 'error' in o['outcome'].lower() or 'fail' in o['outcome'].lower()]
        if len(error_outputs) >= 4:
            return True, f"repeated_errors ({len(error_outputs)}/{len(outputs)})", "rollback_and_rethink"
        
        return False, "", ""
    
    def get_pattern_summary(self) -> str:
        """Get summary of recent patterns for self-reflection."""
        if len(self.recent_outputs) < 3:
            return "Not enough history for pattern analysis."
        
        outputs = list(self.recent_outputs)
        outcomes = [o['outcome'] for o in outputs]
        
        summary = f"Last {len(outputs)} iterations:\n"
        summary += f"- Outcomes: {', '.join(outcomes)}\n"
        
        # Count outcome types
        from collections import Counter
        counts = Counter(outcomes)
        summary += f"- Distribution: {dict(counts)}\n"
        
        return summary
    
    def clear(self):
        """Clear history after recovery."""
        self.recent_outputs.clear()


# =============================================================================
# ADAPTIVE CONTEXT - Smart context selection
# =============================================================================

class AdaptiveContext:
    """
    Intelligently select context based on current task.
    
    Instead of fixed limits, allocate budget by relevance.
    """
    
    def __init__(self, semantic: SemanticSearch):
        self.semantic = semantic
    
    def build_context(self, 
                      current_task: str,
                      mission: str,
                      architecture: str,
                      learnings: str,
                      memory_context: str,
                      guardrails: str,
                      budget_chars: int = CONTEXT_BUDGET_CHARS) -> Dict[str, str]:
        """
        Build optimized context within budget.
        
        Priority order:
        1. Guardrails (critical constraints)
        2. Relevant learnings (errors, decisions)
        3. Architecture (relevant parts)
        4. Memory (relevant history)
        5. Mission (compressed if needed)
        """
        
        result = {}
        remaining = budget_chars
        
        # 1. Guardrails - critical, always include
        if guardrails:
            guardrails_budget = min(len(guardrails), GUARDRAILS_LIMIT)
            result['guardrails'] = guardrails[:guardrails_budget]
            remaining -= len(result['guardrails'])
        
        # 2. Learnings - prioritize error-related
        if learnings and remaining > 20000:
            learnings_budget = min(len(learnings), LEARNINGS_LIMIT, remaining // 3)
            result['learnings'] = self._select_relevant(learnings, current_task, learnings_budget)
            remaining -= len(result['learnings'])
        
        # 3. Architecture - relevant parts
        if architecture and remaining > 15000:
            arch_budget = min(len(architecture), ARCHITECTURE_LIMIT, remaining // 4)
            result['architecture'] = self._select_relevant(architecture, current_task, arch_budget)
            remaining -= len(result['architecture'])
        
        # 4. Memory context
        if memory_context and remaining > 10000:
            mem_budget = min(len(memory_context), MEMORY_CONTEXT_LIMIT, remaining // 4)
            result['memory'] = memory_context[:mem_budget]
            remaining -= len(result['memory'])
        
        # 5. Mission - compress if needed
        if mission and remaining > 5000:
            mission_budget = min(len(mission), MISSION_LIMIT, remaining // 3)
            result['mission'] = mission[:mission_budget]
            remaining -= len(result['mission'])
        
        return result
    
    def _select_relevant(self, content: str, query: str, budget: int) -> str:
        """Select relevant parts of content within budget."""
        if len(content) <= budget:
            return content
        
        # Split into sections
        sections = re.split(r'\n(?=##?\s)', content)
        if len(sections) <= 1:
            return content[:budget]
        
        # Score sections by relevance
        # FIX: Use enumerate to preserve indices correctly (avoids bug with duplicate sections)
        similarities = self.semantic.batch_similarities(query, sections)
        scored = [(i, section, sim) for i, (section, sim) in enumerate(zip(sections, similarities))]
        scored.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
        
        # Build result
        result = []
        chars = 0
        
        for idx, section, score in scored:
            if chars + len(section) > budget:
                continue
            result.append((idx, section))  # Keep original index
            chars += len(section)
        
        # Sort by original order
        result.sort(key=lambda x: x[0])
        return '\n\n'.join(s for _, s in result)


# =============================================================================
# STUCK DETECTOR - Enhanced
# =============================================================================

class StuckDetector:
    """Enhanced stuck detection with better recovery."""
    
    def __init__(self):
        self.stuck_file = RALPH_DIR / "stuck_history.json"
        self.max_task_attempts = 3
        self.max_same_errors = 4
        self.max_no_progress = 8
    
    def check_stuck(self, current_task_id: str, iteration: int) -> Tuple[bool, str]:
        """Check if stuck."""
        history = self._load_history()
        
        # Task attempts
        task_attempts = [h for h in history if h.get('task_id') == current_task_id and not h.get('completed')]
        if len(task_attempts) >= self.max_task_attempts:
            return True, f"Task {current_task_id} failed {len(task_attempts)} times"
        
        # Same error
        # FIX: Filter out empty strings to prevent false positives
        recent_errors = [h.get('error', '') for h in history[-self.max_same_errors:] 
                        if h.get('error') and h.get('error').strip()]
        if len(recent_errors) >= self.max_same_errors:
            unique_errors = set(e[:50] for e in recent_errors)
            if len(unique_errors) == 1:
                return True, f"Same error {len(recent_errors)} times"
        
        # No progress
        recent = history[-self.max_no_progress:]
        if len(recent) >= self.max_no_progress:
            completed = sum(1 for h in recent if h.get('completed'))
            if completed == 0:
                return True, f"No progress in {self.max_no_progress} iterations"
        
        return False, ""
    
    def record_attempt(self, task_id: str, iteration: int, completed: bool,
                       error: Optional[str] = None, attempt_type: str = 'worker'):
        history = self._load_history()
        history.append({
            'task_id': task_id,
            'iteration': iteration,
            'completed': completed,
            'error': error[:200] if error else None,
            'type': attempt_type,
            'timestamp': datetime.now().isoformat()
        })
        history = history[-100:]
        self._save_history(history)
    
    def get_recovery_strategy(self, reason: str) -> str:
        if 'failed' in reason.lower():
            return 'skip_task'
        elif 'error' in reason.lower():
            return 'rollback'
        elif 'progress' in reason.lower():
            return 'escalate'
        return 'skip_task'
    
    def get_task_attempts(self, task_id: str) -> str:
        history = self._load_history()
        attempts = [h for h in history if h.get('task_id') == task_id and not h.get('completed')]
        
        if not attempts:
            return ""
        
        lines = [f"## Previous attempts on {task_id} ({len(attempts)} failures):"]
        for a in attempts[-3:]:
            error = a.get('error', 'unknown')[:100]
            lines.append(f"- Iteration {a['iteration']}: {error}")
        return '\n'.join(lines)
    
    def clear_history(self):
        self._save_history([])
    
    def _load_history(self) -> List[dict]:
        return safe_read_json(self.stuck_file, [])
    
    def _save_history(self, history: List[dict]):
        atomic_write_json(self.stuck_file, history)


# =============================================================================
# METRICS
# =============================================================================

class RalphMetrics:
    """Track performance metrics."""
    
    def __init__(self):
        self.metrics_file = RALPH_DIR / "metrics.json"
        self._metrics = self._load()
    
    def _load(self) -> dict:
        return safe_read_json(self.metrics_file, {
            "total_iterations": 0,
            "successful": 0,
            "failed": 0,
            "stuck_count": 0,
            "condense_count": 0,
            "reflection_count": 0,
            "divergence_detected": 0,
            "iteration_times": [],
            "started_at": None
        })
    
    def _save(self):
        self._metrics["last_updated"] = datetime.now().isoformat()
        atomic_write_json(self.metrics_file, self._metrics)
    
    def record_iteration(self, success: bool, time_seconds: float, stuck: bool = False):
        if self._metrics["started_at"] is None:
            self._metrics["started_at"] = datetime.now().isoformat()
        
        self._metrics["total_iterations"] += 1
        if success:
            self._metrics["successful"] += 1
        else:
            self._metrics["failed"] += 1
        if stuck:
            self._metrics["stuck_count"] += 1
        
        self._metrics["iteration_times"].append(time_seconds)
        self._metrics["iteration_times"] = self._metrics["iteration_times"][-50:]
        self._save()
    
    def record_condense(self):
        self._metrics["condense_count"] += 1
        self._save()
    
    def record_reflection(self):
        self._metrics["reflection_count"] += 1
        self._save()
    
    def record_divergence(self):
        self._metrics["divergence_detected"] += 1
        self._save()


# =============================================================================
# DISK MONITOR
# =============================================================================

class DiskSpaceMonitor:
    """Monitor and cleanup disk space."""
    
    def get_free_mb(self) -> float:
        try:
            stat = os.statvfs(RALPH_DIR)
            return (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        except Exception:
            return float('inf')
    
    def is_low(self) -> bool:
        return self.get_free_mb() < 500
    
    def cleanup(self, emergency: bool = False):
        """Clean up old files.
        
        FIX: Uses modification time for sorting and protects files newer than 1 hour.
        This prevents deleting files from the current iteration batch.
        
        With git-native mode (v3.0), most files are in git, so only log truncation
        is performed. Iteration history is in git commits.
        """
        stats = {"deleted": 0, "truncated": 0}
        
        # Git-native mode: minimal cleanup (only truncate logs)
        if GIT_STATE_AVAILABLE:
            max_size = 300_000 if emergency else 500_000
            if LOG_FILE.exists() and LOG_FILE.stat().st_size > max_size:
                try:
                    content = LOG_FILE.read_text()
                    LOG_FILE.write_text(f"...(rotated)...\n{content[-max_size:]}")
                    stats["truncated"] += 1
                    log(f"Git-native cleanup: truncated log")
                except Exception:
                    pass
            return
        
        # Truncate log
        max_size = 300_000 if emergency else 500_000
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > max_size:
            try:
                content = LOG_FILE.read_text()
                LOG_FILE.write_text(f"...(rotated)...\n{content[-max_size:]}")
                stats["truncated"] += 1
            except Exception:
                pass
        
        # Delete old iteration logs
        keep = 100 if emergency else MAX_ITERATION_LOGS
        cutoff_time = time.time() - 3600  # 1 hour ago
        
        if ITERATIONS_DIR.exists():
            # Sort by modification time (oldest first), not alphabetically
            log_files = sorted(ITERATIONS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
            for old in log_files[:-keep]:
                try:
                    # FIX: Don't delete files newer than 1 hour
                    if old.stat().st_mtime < cutoff_time:
                        old.unlink()
                        stats["deleted"] += 1
                except Exception:
                    pass
            
            json_files = sorted(ITERATIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
            for old in json_files[:-keep]:
                try:
                    if old.stat().st_mtime < cutoff_time:
                        old.unlink()
                        stats["deleted"] += 1
                except Exception:
                    pass
        
        if stats["deleted"] or stats["truncated"]:
            log(f"Cleanup: deleted={stats['deleted']}, truncated={stats['truncated']}")


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """Circuit breaker for external services.
    
    FIX: State is now persisted to survive daemon restarts.
    This prevents crash-loop bypass where failures reset on restart.
    """
    
    def __init__(self, name: str, threshold: int = 5, timeout: float = 120.0):
        self.name = name
        self.threshold = threshold
        self.timeout = timeout
        self.state_file = RALPH_DIR / f"circuit_{name}.json"
        
        # Load persisted state
        state = self._load_state()
        self.failures = state.get('failures', 0)
        self.state = state.get('state', 'CLOSED')
        self.last_failure = state.get('last_failure', 0.0)
    
    def _load_state(self) -> dict:
        """Load circuit breaker state from disk."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {}
    
    def _save_state(self):
        """Persist circuit breaker state to disk."""
        try:
            atomic_write_json(self.state_file, {
                'failures': self.failures,
                'state': self.state,
                'last_failure': self.last_failure
            })
        except Exception:
            pass
    
    def can_proceed(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            # FIX: Use max(0, ...) to handle clock jumps (NTP sync can make time go backward)
            elapsed = max(0, time.time() - self.last_failure)
            if elapsed >= self.timeout:
                self.state = "HALF_OPEN"
                self._save_state()
                return True
            return False
        return True
    
    def record_success(self):
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            log(f"Circuit {self.name}: recovered")
        self.failures = 0
        self._save_state()
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.threshold:
            self.state = "OPEN"
            log(f"Circuit {self.name}: OPEN")
        self._save_state()


# =============================================================================
# EPOCH MANAGER
# =============================================================================

class EpochManager:
    """Manage milestone snapshots."""
    
    def __init__(self):
        self.epochs_dir = RALPH_DIR / "epochs"
        self.epochs_dir.mkdir(exist_ok=True)
    
    def check_milestone(self, iteration: int) -> Optional[str]:
        config = read_config()
        last_epoch = config.get("lastEpochIteration", 0)
        
        if iteration - last_epoch < 25:
            return None
        
        prd = read_prd()
        stories = prd.get("userStories", [])
        done = sum(1 for s in stories if s.get("passes"))
        total = len(stories) or 1
        
        if iteration % 100 == 0:
            return f"iteration_{iteration}"
        
        pct = done / total
        if pct >= 0.5 and not config.get("50pctSaved"):
            return "50pct_complete"
        if pct >= 0.75 and not config.get("75pctSaved"):
            return "75pct_complete"
        
        return None
    
    def save_epoch(self, iteration: int, reason: str):
        config = read_config()
        epoch_num = config.get("currentEpoch", 0) + 1
        
        prd = read_prd()
        stories = prd.get("userStories", [])
        done = sum(1 for s in stories if s.get("passes"))
        
        epoch_data = {
            "epoch": epoch_num,
            "iteration": iteration,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "progress": {"done": done, "total": len(stories)}
        }
        
        epoch_file = self.epochs_dir / f"epoch_{epoch_num:03d}.json"
        atomic_write_json(epoch_file, epoch_data)
        
        update_config("currentEpoch", epoch_num)
        update_config("lastEpochIteration", iteration)
        
        if "50pct" in reason:
            update_config("50pctSaved", True)
        if "75pct" in reason:
            update_config("75pctSaved", True)
        
        log(f"Saved epoch {epoch_num}: {reason}")


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

semantic_search: SemanticSearch = None
memory: HierarchicalMemory = None
learnings_mgr: LearningsManager = None
condenser: ContextCondenser = None
divergence_detector: DivergenceDetector = None
adaptive_context: AdaptiveContext = None
metrics: RalphMetrics = None
epochs: EpochManager = None
stuck_detector: StuckDetector = None
circuit_breaker: CircuitBreaker = None
disk_monitor: DiskSpaceMonitor = None


def init_managers():
    """Initialize all managers."""
    global semantic_search, memory, learnings_mgr, condenser
    global divergence_detector, adaptive_context
    global metrics, epochs, stuck_detector, circuit_breaker, disk_monitor
    
    semantic_search = SemanticSearch()
    memory = HierarchicalMemory(semantic_search)
    learnings_mgr = LearningsManager(semantic_search)
    condenser = ContextCondenser(semantic_search)
    divergence_detector = DivergenceDetector(semantic_search)
    adaptive_context = AdaptiveContext(semantic_search)
    metrics = RalphMetrics()
    epochs = EpochManager()
    stuck_detector = StuckDetector()
    circuit_breaker = CircuitBreaker("openhands", threshold=5, timeout=120)
    disk_monitor = DiskSpaceMonitor()
    
    log("All managers initialized")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_current_task(prd: dict) -> Optional[dict]:
    """Get next pending task."""
    for story in prd.get("userStories", []):
        if not story.get("passes", False) and not story.get("blocked", False):
            return story
    return None


def mark_task_done(task_id: str, passed: bool = True):
    """Mark a task as done with proper file locking to prevent concurrent write corruption.
    
    CRITICAL FIX: Uses fcntl.flock() for exclusive file locking instead of
    optimistic locking. The original implementation had a TOCTOU race condition
    where another process could write between version check and save.
    
    File locking ensures atomic read-modify-write operations.
    
    HIGH-5 FIX: Also updates TaskManager when git state is available.
    """
    # Update TaskManager first (git-native path)
    if GIT_STATE_AVAILABLE and _task_manager:
        status = "done" if passed else "failed"
        if _task_manager.set_task_status(task_id, status):
            log(f"Task {task_id} marked {status} in TaskManager")
        else:
            log_warning(f"Failed to update task {task_id} in TaskManager")
    
    # Also update legacy PRD for backward compatibility
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Open for read+write, create if not exists
            with open(PRD_FILE, 'r+') as f:
                # Acquire exclusive lock (blocking)
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    # Read while holding lock
                    content = f.read()
                    prd = json.loads(content) if content.strip() else {"userStories": [], "verified": False}
                    
                    # Find and update the task
                    task_found = False
                    for story in prd.get("userStories", []):
                        if story.get("id") == task_id:
                            story["passes"] = passed
                            story["completedAt"] = datetime.now().isoformat()
                            task_found = True
                            break
                    
                    if not task_found:
                        log_error(f"Task {task_id} not found in PRD")
                        return
                    
                    # Write back while still holding lock (atomic)
                    f.seek(0)
                    f.truncate()
                    json.dump(prd, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data hits disk
                    
                    log(f"Task {task_id}: {'PASS' if passed else 'FAIL'}")
                    return
                finally:
                    # Release lock (also released automatically on close)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    
        except FileNotFoundError:
            log_error(f"PRD file not found: {PRD_FILE}")
            return
        except json.JSONDecodeError as e:
            log_warning(f"PRD JSON decode error, retrying ({attempt+1}/{max_retries}): {e}")
            time.sleep(0.1 * (attempt + 1))
        except (IOError, OSError) as e:
            log_warning(f"PRD file lock failed, retrying ({attempt+1}/{max_retries}): {e}")
            time.sleep(0.1 * (attempt + 1))
    
    log_error(f"Failed to mark task {task_id} after {max_retries} retries")


def parse_ralph_tags(output: str) -> dict:
    """Parse ralph tags from output."""
    result = {
        "task_done": None,
        "task_verified": None,
        "stuck": False,
        "project_verified": False,
        "learnings": []
    }
    
    match = re.search(r'<ralph>TASK_DONE:([^<]+)</ralph>', output)
    if match:
        result["task_done"] = match.group(1).strip()
    
    match = re.search(r'<ralph>TASK_VERIFIED:([^:]+):(\w+)(?::([^<]+))?</ralph>', output)
    if match:
        result["task_verified"] = {
            "task_id": match.group(1).strip(),
            "passed": match.group(2).upper() == "PASS",
            "reason": match.group(3).strip() if match.group(3) else None
        }
    
    if '<ralph>STUCK</ralph>' in output:
        result["stuck"] = True
    
    if '<ralph>PROJECT_VERIFIED</ralph>' in output:
        result["project_verified"] = True
    
    for match in re.finditer(r'<ralph>LEARNING:([^<]+)</ralph>', output):
        result["learnings"].append(match.group(1).strip())
    
    return result


def read_file_limited(filepath: Path, max_chars: int = 50000) -> str:
    """Read file with limit."""
    if not filepath.exists():
        return ""
    try:
        content = filepath.read_text()
        if len(content) > max_chars:
            return f"...(truncated)...\n{content[-max_chars:]}"
        return content
    except Exception:
        return ""


def get_iteration_type(config: dict) -> str:
    """Determine iteration type."""
    prd = read_prd()
    iteration = config.get("currentIteration", 0)
    
    # Pending verification
    pending = config.get("pendingTaskVerification")
    if pending and pending.get("taskId"):
        return "task_verify"
    
    phase = prd.get("phase", "planning")
    
    if phase == "planning":
        for story in prd.get("userStories", []):
            if story.get("id") == "PLAN" and story.get("passes"):
                prd["phase"] = "execution"
                save_prd(prd)
                return "worker"
        return "planning"
    
    if phase == "verification":
        return "verification"
    
    # Check for condense
    should_cond, reason = condenser.should_condense(iteration, config) if condenser else (False, "")
    if should_cond:
        return "condense"
    
    # Architect interval
    # FIX: Track last architect iteration to prevent duplicate on resume
    arch_interval = config.get("architectInterval", 10)
    last_architect = config.get("lastArchitectIteration", 0)
    if arch_interval > 0 and iteration > 0 and iteration > last_architect and iteration % arch_interval == 0:
        return "architect"
    
    # All done?
    if not get_current_task(prd):
        prd["phase"] = "verification"
        save_prd(prd)
        return "verification"
    
    return "worker"


def build_prompt(config: dict, iter_type: str) -> str:
    """Build prompt with adaptive context."""
    prd = read_prd()
    task = get_current_task(prd)
    iteration = config.get("currentIteration", 0)
    
    task_id = task.get("id", "NONE") if task else "NONE"
    task_title = task.get("title", "") if task else ""
    task_desc = task.get("description", "") if task else ""
    current_task_context = f"{task_title} {task_desc}"
    
    # Load raw content
    mission = read_file_limited(MISSION_FILE, MISSION_LIMIT)
    architecture = read_file_limited(ARCHITECTURE_FILE, ARCHITECTURE_LIMIT)
    guardrails = read_file_limited(RALPH_DIR / "guardrails.md", GUARDRAILS_LIMIT)
    
    # Get relevant learnings
    learnings = ""
    if learnings_mgr:
        learnings = learnings_mgr.get_relevant(current_task_context, LEARNINGS_LIMIT)
    
    # Get memory context
    memory_context = ""
    if memory:
        memory_context = memory.get_context(current_task_context, MEMORY_CONTEXT_LIMIT)
    
    # Use adaptive context to optimize
    if adaptive_context:
        optimized = adaptive_context.build_context(
            current_task=current_task_context,
            mission=mission,
            architecture=architecture,
            learnings=learnings,
            memory_context=memory_context,
            guardrails=guardrails
        )
        mission = optimized.get('mission', mission)
        architecture = optimized.get('architecture', architecture)
        learnings = optimized.get('learnings', learnings)
        memory_context = optimized.get('memory', memory_context)
        guardrails = optimized.get('guardrails', guardrails)
    
    # Load template
    template = load_prompt_template(iter_type)
    
    # Substitutions
    stories = prd.get("userStories", [])
    done = sum(1 for s in stories if s.get("passes"))
    
    pending = config.get("pendingTaskVerification", {}) or {}
    
    subs = {
        "${iteration}": str(iteration),
        "${task_id}": task_id,
        "${task_title}": task_title,
        "${task_description}": task_desc,
        "${done_tasks}": str(done),
        "${total_tasks}": str(len(stories)),
        "${mission}": mission,
        "${learnings}": learnings,
        "${architecture}": architecture,
        "${memory_context}": memory_context,
        "${guardrails_section}": guardrails,
        "${verify_task_id}": pending.get("taskId", task_id),
        "${previous_attempts}": stuck_detector.get_task_attempts(task_id) if stuck_detector else "",
    }
    
    result = template
    for var, val in subs.items():
        result = result.replace(var, val)
    
    return result


def load_prompt_template(iter_type: str) -> str:
    """Load prompt template."""
    # Project-specific
    project_prompt = RALPH_DIR / "prompts" / f"{iter_type}.md"
    if project_prompt.exists():
        return project_prompt.read_text()
    
    # Global
    global_prompt = RALPH_DIR / "prompts" / "global" / f"{iter_type}.md"
    if global_prompt.exists():
        return global_prompt.read_text()
    
    # Fallbacks
    fallbacks = {
        "planning": "# Planning - Iter ${iteration}\nCreate execution plan. <ralph>TASK_DONE:PLAN</ralph> when done.",
        "worker": "# Worker - Iter ${iteration}\nTask: ${task_id} - ${task_title}\n${task_description}\n\nWhen done: <ralph>TASK_DONE:${task_id}</ralph>",
        "task_verify": "# Verify - Iter ${iteration}\nVerify ${verify_task_id}.\n<ralph>TASK_VERIFIED:${verify_task_id}:PASS</ralph> or FAIL:reason",
        "architect": "# Architect - Iter ${iteration}\nReview architecture. Progress: ${done_tasks}/${total_tasks}",
        "verification": "# Project Verification\n<ralph>PROJECT_VERIFIED</ralph> when complete.",
        "condense": "# Condense Context\nSummarize key learnings. Remove noise."
    }
    
    return fallbacks.get(iter_type, f"Continue. Type: {iter_type}")


def handle_iteration_result(iteration: int, iter_type: str, output: str, config: dict) -> str:
    """Handle iteration result."""
    tags = parse_ralph_tags(output)
    result = "unknown"
    
    # Save learnings
    if tags["learnings"] and learnings_mgr:
        for learning in tags["learnings"]:
            learnings_mgr.add(learning, iteration)
            # Also save to git notes (v3.0)
            add_learning_to_git(f"[iter-{iteration}] {learning}")
    
    # Record in divergence detector - store iteration number for later update
    divergence_iteration = None
    if divergence_detector:
        summary = output[:500] if len(output) > 500 else output
        divergence_iteration = divergence_detector.record_iteration(iteration, summary, result)
    
    # Handle by type
    if iter_type == "task_verify":
        if tags["task_verified"]:
            tv = tags["task_verified"]
            if tv["passed"]:
                mark_task_done(tv["task_id"], True)
                config.pop("pendingTaskVerification", None)
                save_config(config)
                result = "verified_pass"
            else:
                config.pop("pendingTaskVerification", None)
                save_config(config)
                result = "verified_fail"
        else:
            result = "verify_retry"
    
    elif iter_type == "worker":
        if tags["task_done"]:
            task_id = tags["task_done"]
            if config.get("requireVerification", True):
                prd = read_prd()
                task = None
                for s in prd.get("userStories", []):
                    if s.get("id") == task_id:
                        task = s
                        break
                config["pendingTaskVerification"] = {
                    "taskId": task_id,
                    "taskTitle": task.get("title", "") if task else "",
                    "taskDescription": task.get("description", "") if task else ""
                }
                save_config(config)
                result = "pending_verify"
            else:
                mark_task_done(task_id, True)
                result = "task_done"
        elif tags["stuck"]:
            result = "stuck"
        else:
            result = "working"
    
    elif iter_type == "planning":
        if tags["task_done"] and "PLAN" in tags["task_done"].upper():
            mark_task_done("PLAN", True)
            result = "planning_done"
        else:
            result = "planning"
    
    elif iter_type == "verification":
        if tags["project_verified"]:
            prd = read_prd()
            prd["verified"] = True
            save_prd(prd)
            update_config("status", "complete")
            result = "project_verified"
        else:
            result = "verification_pending"
    
    elif iter_type == "architect":
        # FIX: Track last architect iteration to prevent duplicate on resume
        # NOTE: Use the iteration parameter, not config value (which is not yet persisted)
        update_config("lastArchitectIteration", iteration)
        result = "architect_done"
    
    elif iter_type == "condense":
        if condenser:
            condensed = condenser.parse_condensed(output)
            if condensed:
                original = learnings_mgr.get_all() if learnings_mgr else ""
                condenser.save_condensed(iteration, condensed, original)
                result = "condense_done"
            else:
                result = "condense_failed"
        else:
            result = "condense_done"
    
    # Update divergence detector with final result
    # FIX: Find entry by iteration number instead of using reference (avoids invalidation bug)
    if divergence_iteration is not None and divergence_detector:
        for entry in divergence_detector.recent_outputs:
            if entry['iteration'] == divergence_iteration:
                entry['outcome'] = result
                break
    
    return result


def run_iteration(config: dict) -> Tuple[bool, str]:
    """Run single iteration.
    
    FIX: Iteration counter is now incremented AFTER successful completion,
    not at the start. This prevents skipped iterations if daemon crashes mid-iteration.
    """
    global current_process
    
    # Calculate next iteration but don't persist yet
    iteration = config.get("currentIteration", 0) + 1
    # NOTE: We'll persist this AFTER successful completion (see end of function)
    
    if circuit_breaker and not circuit_breaker.can_proceed():
        log_warning("Circuit breaker OPEN")
        time.sleep(30)
        return False, "circuit_open"
    
    iter_type = get_iteration_type(config)
    log(f"=== Iteration {iteration} ({iter_type}) ===")
    
    # Check divergence
    if divergence_detector and iter_type == "worker":
        is_div, reason, action = divergence_detector.check_divergence()
        if is_div:
            log_warning(f"DIVERGENCE: {reason}")
            if metrics:
                metrics.record_divergence()
            
            if action == "escalate_to_architect":
                iter_type = "architect"
            elif action == "force_new_approach":
                # Add to learnings
                if learnings_mgr:
                    learnings_mgr.add(f"DIVERGENCE DETECTED: {reason}. Need fresh approach.", iteration)
    
    # Check stuck
    prd = read_prd()
    current_task = get_current_task(prd)
    if stuck_detector and current_task and iter_type == "worker":
        task_id = current_task.get("id", "")
        is_stuck, reason = stuck_detector.check_stuck(task_id, iteration)
        if is_stuck:
            log_warning(f"STUCK: {reason}")
            strategy = stuck_detector.get_recovery_strategy(reason)
            if strategy == "skip_task":
                current_task["blocked"] = True
                current_task["blockedReason"] = reason
                save_prd(prd)
                stuck_detector.clear_history()
                return True, "task_skipped"
            elif strategy == "escalate":
                iter_type = "architect"
    
    # Build prompt
    prompt = build_prompt(config, iter_type)
    
    # Create output file
    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = ITERATIONS_DIR / f"iteration_{iteration:04d}.log"
    
    # SECURITY FIX: Pass arguments directly to avoid shell injection
    # (previously used bash -c which had double-interpretation risk)
    
    start_time = time.time()
    
    try:
        with open(output_file, "w") as f:
            # FIX: Use lock when setting current_process for thread-safe signal handling
            # SECURITY FIX: Direct argument passing, no shell=True or bash -c
            proc = subprocess.Popen(
                ["openhands", "--headless", "--json", "-t", prompt],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd="/workspace"
            )
            # FIX: Removed redundant 'global current_process' from nested block
            # (already declared at function level)
            if _process_lock:
                with _process_lock:
                    current_process = proc
            else:
                current_process = proc
            
            # SECURITY FIX: Cap timeout at DOCKER_TIMEOUT (8 hours) to prevent
            # unbounded wait if config is missing or corrupt
            timeout = min(config.get("sessionTimeoutSeconds", 1800), DOCKER_TIMEOUT)
            try:
                proc.wait(timeout=timeout)  # FIX: Removed unused exit_code assignment
            except subprocess.TimeoutExpired:
                log_warning(f"Timeout after {timeout}s")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # FIX: Force kill if terminate doesn't work
                    proc.kill()
                    proc.wait()
                return False, "timeout"
        
        # FIX: Clear current_process with lock
        if _process_lock:
            with _process_lock:
                current_process = None
        else:
            current_process = None
        elapsed = time.time() - start_time
        
        # SECURITY FIX: Limit output file size to prevent OOM
        MAX_OUTPUT_SIZE = 10_000_000  # 10MB
        output = ""
        if output_file.exists():
            size = output_file.stat().st_size
            if size > MAX_OUTPUT_SIZE:
                log_warning(f"Output file too large ({size} bytes), truncating to {MAX_OUTPUT_SIZE}")
                with open(output_file, 'r', errors='replace') as f:
                    output = f.read(MAX_OUTPUT_SIZE)
            else:
                output = output_file.read_text(errors='replace')
        
        result = handle_iteration_result(iteration, iter_type, output, config)
        
        # Save metadata
        meta = {
            "iteration": iteration,
            "type": iter_type,
            "result": result,
            "duration": int(elapsed),
            "timestamp": datetime.now().isoformat()
        }
        atomic_write_json(ITERATIONS_DIR / f"iteration_{iteration:04d}.json", meta)
        
        # Update memory
        if memory and current_task:
            task_id = current_task.get("id", iter_type)
            tags = parse_ralph_tags(output)
            memory.add_iteration(
                iteration=iteration,
                task_id=task_id,
                summary=f"{iter_type}: {result}",
                details=output[-2000:],
                key_points=tags["learnings"][:5],
                outcome=result
            )
        
        # Metrics
        if metrics:
            metrics.record_iteration(True, elapsed, result == "stuck")
            if result == "condense_done":
                metrics.record_condense()
        
        # Stuck detector
        if stuck_detector and current_task:
            task_id = current_task.get("id", "")
            completed = result in ["task_done", "pending_verify", "verified_pass"]
            stuck_detector.record_attempt(task_id, iteration, completed)
        
        # Circuit breaker
        if circuit_breaker:
            circuit_breaker.record_success()
        
        # Epoch
        if epochs:
            milestone = epochs.check_milestone(iteration)
            if milestone:
                epochs.save_epoch(iteration, milestone)
        
        # FIX: Only increment counter AFTER successful completion
        # This prevents lost iterations if daemon crashes mid-iteration
        update_config("currentIteration", iteration)
        
        # Git-native state tracking (v3.0)
        task_id = current_task.get("id", "") if current_task else iter_type
        set_git_iteration(iteration)
        set_git_task(task_id)
        
        # Commit iteration to git with structured message
        summary = f"{result}"[:50]
        commit_iteration_to_git(iteration, task_id, summary)
        
        # Save checkpoint for recovery
        save_checkpoint_to_git(iteration, task_id, "completed" if result in ["task_done", "verified_pass"] else "in_progress")
        
        log(f"Iteration {iteration}: {result} ({int(elapsed)}s)")
        return True, result
        
    except Exception as e:
        log_error(f"Iteration failed: {e}")
        log_error(traceback.format_exc())
        
        # FIX: Clear current_process with lock
        if _process_lock:
            with _process_lock:
                current_process = None
        else:
            current_process = None
        
        if metrics:
            metrics.record_iteration(False, 0)
        if circuit_breaker:
            circuit_breaker.record_failure()
        
        return False, "error"


def main():
    """Main daemon loop."""
    global _process_lock
    
    # FIX: Initialize process lock for thread-safe signal handling
    _process_lock = threading.Lock()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Ensure directories
    # FIX: Use 0o755 instead of 0o777 for better security (owner rwx, group/other rx)
    for d in [RALPH_DIR, ITERATIONS_DIR, MEMORY_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, 0o755)
        except Exception:
            pass
    
    # SECURITY FIX: Write PID file with exclusive lock to prevent impersonation
    try:
        # Try to create with O_EXCL (fails if exists = atomic check-and-create)
        fd = os.open(str(PID_FILE), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
    except FileExistsError:
        # Check if existing daemon is actually running
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, 0)  # Check if process exists
            log_error(f"Another daemon already running (PID {old_pid})")
            sys.exit(1)
        except (ProcessLookupError, ValueError, FileNotFoundError):
            # Stale PID file, safe to overwrite
            try:
                PID_FILE.unlink()
            except FileNotFoundError:
                pass
            fd = os.open(str(PID_FILE), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
    
    log("=" * 60)
    log(f"Ralph Daemon v{DAEMON_VERSION} Started - Git-Native State")
    log(f"PID: {os.getpid()}")
    log(f"Context budget: {CONTEXT_BUDGET_CHARS} chars (~{CONTEXT_BUDGET_CHARS // 4000}K tokens)")
    log("=" * 60)
    
    init_managers()
    
    consecutive_errors = 0
    retry_delay = BASE_DELAY
    
    # CRITICAL FIX: Track iterations per minute to prevent infinite loops
    iteration_times: deque = deque(maxlen=60)  # Last 60 iteration timestamps
    
    try:
        while not _shutdown_event.is_set():
            write_heartbeat()
            
            config = read_config()
            status = config.get("status", "paused")
            
            if status == "starting":
                update_config("status", "running")
                status = "running"
            
            if status == "running":
                current_iter = config.get("currentIteration", 0)
                
                # CRITICAL FIX: Rate limiting - force pause if >60 iterations/minute
                now = time.time()
                iteration_times.append(now)
                if len(iteration_times) >= 60:
                    # Check if 60 iterations happened in the last 60 seconds
                    oldest = iteration_times[0]
                    if now - oldest < 60:
                        log_error("CRITICAL: >60 iterations/minute detected! Forcing pause to prevent infinite loop.")
                        update_config("status", "error")
                        update_config("lastError", "Too many iterations per minute - possible infinite loop")
                        time.sleep(60)  # Force long pause
                        iteration_times.clear()  # Reset after pause
                        continue
                
                # Disk check
                if disk_monitor and disk_monitor.is_low():
                    disk_monitor.cleanup(emergency=True)
                
                # Periodic maintenance
                if current_iter > 0 and current_iter % COMPACT_INTERVAL == 0:
                    if disk_monitor:
                        disk_monitor.cleanup()
                    if memory and semantic_search:
                        memory.compact(semantic_search)
                    if semantic_search:
                        semantic_search.flush_cache()
                
                # Max iterations
                # FIX: Validate maxIterations is a valid positive integer
                max_iter = config.get("maxIterations", 0)
                if not isinstance(max_iter, int) or max_iter < 0:
                    max_iter = 0  # Treat invalid as unlimited
                if max_iter > 0 and current_iter >= max_iter:
                    log(f"Reached max iterations ({max_iter})")
                    update_config("status", "complete")
                    continue
                
                # Already complete?
                prd = read_prd()
                if prd.get("verified"):
                    update_config("status", "complete")
                    continue
                
                # Run iteration
                success, result = run_iteration(config)
                
                if success:
                    consecutive_errors = 0
                    retry_delay = BASE_DELAY
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        update_config("status", "error")
                        consecutive_errors = 0
                    else:
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 300)
                        continue
                
                # Pause between
                pause = config.get("pauseBetweenSeconds", 10)
                if pause > 0:
                    time.sleep(pause)
            
            elif status in ["paused", "needs_help"]:
                time.sleep(5 if status == "paused" else 30)
            
            elif status in ["stopped", "complete", "error"]:
                time.sleep(10)
            
            else:
                time.sleep(5)
    
    except KeyboardInterrupt:
        log("Keyboard interrupt")
    except Exception as e:
        log_error(f"Fatal error: {e}")
        log_error(traceback.format_exc())
    finally:
        log("Ralph daemon shutting down")
        try:
            PID_FILE.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
