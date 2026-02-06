#!/usr/bin/env python3
"""
Ralph Daemon - Full-featured autonomous development daemon.

Runs inside Docker container with complete logic:
- HierarchicalMemory (hot/warm/cold tiers)
- ContextCondenser with verification
- LearningsManager with semantic deduplication
- SemanticSearch using sentence-transformers
- StuckDetector with recovery strategies
- CircuitBreaker for service resilience
- DiskSpaceMonitor with cleanup
- Full prompt variable substitution
- Robust error handling with retries
- Watchdog-compatible heartbeat

Auto-installs dependencies if missing.

Usage (inside container):
    python3 /workspace/.ralph/ralph_daemon.py
"""

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
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


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
# CONSTANTS
# =============================================================================

RALPH_DIR = Path("/workspace/.ralph")
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

# Limits
MAX_RETRIES = 3
BASE_DELAY = 30
MAX_CONSECUTIVE_ERRORS = 5
CONTEXT_SIZE_LIMIT = 150000  # chars, ~37K tokens
LEARNINGS_LIMIT = 30000
ARCHITECTURE_LIMIT = 10000
MISSION_LIMIT = 10000
MEMORY_CONTEXT_LIMIT = 30000

# Semantic thresholds
SEMANTIC_DUPLICATE_THRESHOLD = 0.55  # For learnings deduplication
SEMANTIC_COMPACT_THRESHOLD = 0.70    # For cold memory compaction
SEMANTIC_RELEVANCE_THRESHOLD = 0.15  # For selecting relevant content

# Verification
MAX_VERIFY_RETRIES = 3  # Retries if no TASK_VERIFIED tag

# Daemon state
shutdown_requested = False
current_process = None


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


# =============================================================================
# ATOMIC FILE OPERATIONS
# =============================================================================

def atomic_write_text(filepath: Path, content: str) -> bool:
    """
    Write text to file, preserving ownership if file exists.
    
    Strategy to avoid permission issues between Docker (root) and host:
    - If file exists: write directly (preserves original ownership)
    - If file doesn't exist: create with 0666 permissions
    
    This allows host-created files to remain writable by host after daemon updates them.
    """
    filepath = Path(filepath)
    
    try:
        # Ensure parent exists with open permissions
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(filepath.parent, 0o777)
        except Exception:
            pass
        
        file_existed = filepath.exists()
        
        # Write content directly (preserves ownership if file exists)
        filepath.write_text(content)
        
        # Only chmod new files (existing files keep their ownership/permissions)
        if not file_existed:
            try:
                os.chmod(filepath, 0o666)
            except Exception:
                pass
        
        return True
    except Exception as e:
        log_error(f"Write failed for {filepath}: {e}")
        return False


def atomic_write_json(filepath: Path, data: dict) -> bool:
    """Atomically write JSON file."""
    return atomic_write_text(filepath, json.dumps(data, indent=2, ensure_ascii=False))


# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested, current_process
    log(f"Received signal {signum}, shutting down...")
    shutdown_requested = True
    if current_process and current_process.poll() is None:
        log("Terminating current iteration...")
        current_process.terminate()
        try:
            current_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            current_process.kill()


def write_heartbeat():
    """Write heartbeat timestamp."""
    try:
        HEARTBEAT_FILE.write_text(str(int(time.time())))
        os.chmod(HEARTBEAT_FILE, 0o666)
    except Exception as e:
        log_error(f"Failed to write heartbeat: {e}")


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
    """Read Ralph config."""
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
    except Exception as e:
        log_error(f"Failed to read config: {e}")
    return {"status": "paused"}


def save_config(config: dict):
    """Save full config atomically."""
    if not atomic_write_json(CONFIG_FILE, config):
        log_error("Failed to save config")


def update_config(key: str, value):
    """Update single config value."""
    config = read_config()
    config[key] = value
    save_config(config)


def read_prd() -> dict:
    """Read PRD file."""
    try:
        if PRD_FILE.exists():
            return json.loads(PRD_FILE.read_text())
    except Exception as e:
        log_error(f"Failed to read PRD: {e}")
    return {"userStories": [], "verified": False}


def save_prd(prd: dict):
    """Save PRD file atomically."""
    if not atomic_write_json(PRD_FILE, prd):
        log_error("Failed to save PRD")


# =============================================================================
# SEMANTIC SEARCH - Using sentence-transformers for true semantic similarity
# =============================================================================

class SemanticSearch:
    """
    Semantic search using Sentence Transformers embeddings.
    Uses all-mpnet-base-v2 model for best quality English text similarity.
    """
    
    _model = None
    _model_loaded = False
    
    def __init__(self):
        self.cache_file = RALPH_DIR / "semantic_cache.json"
        self._embeddings_cache = {}
        self._load_cache()
    
    @classmethod
    def _get_model(cls):
        """Lazy-load Sentence Transformer model (once per process)."""
        if cls._model_loaded:
            return cls._model
        
        cls._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer
            # Use best quality model for English
            cls._model = SentenceTransformer('all-mpnet-base-v2')
            log("Loaded sentence-transformers model: all-mpnet-base-v2")
        except ImportError:
            log_error("sentence-transformers not installed! Run: pip install sentence-transformers")
            raise RuntimeError("sentence-transformers required for semantic search")
        except Exception as e:
            log_error(f"Failed to load sentence-transformers model: {e}")
            raise
        return cls._model
    
    def _load_cache(self):
        if self.cache_file.exists():
            try:
                self._embeddings_cache = json.loads(self.cache_file.read_text())
            except Exception:
                self._embeddings_cache = {}
    
    def _save_cache(self):
        try:
            # Limit cache to 10000 entries
            if len(self._embeddings_cache) > 10000:
                keys = list(self._embeddings_cache.keys())[-10000:]
                self._embeddings_cache = {k: self._embeddings_cache[k] for k in keys}
            self.cache_file.write_text(json.dumps(self._embeddings_cache))
            os.chmod(self.cache_file, 0o666)
        except Exception:
            pass
    
    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _get_embedding(self, text: str):
        """Get embedding for text with caching."""
        import numpy as np
        model = self._get_model()
        
        text_hash = self._text_hash(text)
        if text_hash in self._embeddings_cache:
            return np.array(self._embeddings_cache[text_hash], dtype=np.float32)
        
        embedding = model.encode(text, convert_to_numpy=True)
        self._embeddings_cache[text_hash] = embedding.tolist()
        return embedding.astype(np.float32)
    
    def _get_embeddings_batch(self, texts: List[str]):
        """Get embeddings for multiple texts efficiently."""
        import numpy as np
        model = self._get_model()
        
        results = []
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            if text_hash in self._embeddings_cache:
                results.append((i, np.array(self._embeddings_cache[text_hash], dtype=np.float32)))
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        if texts_to_encode:
            new_embeddings = model.encode(texts_to_encode, convert_to_numpy=True)
            for idx, text, emb in zip(indices_to_encode, texts_to_encode, new_embeddings):
                text_hash = self._text_hash(text)
                self._embeddings_cache[text_hash] = emb.tolist()
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
        """Find most similar documents to query."""
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
    
    def is_duplicate(self, new_text: str, existing_texts: List[str], threshold: float = 0.55) -> bool:
        """Check if new_text is semantically duplicate of any existing text."""
        if not existing_texts:
            return False
        similar = self.find_similar(new_text, existing_texts, top_k=1, threshold=threshold)
        return len(similar) > 0
    
    def select_relevant(self, query: str, sections: List[str], max_chars: int = 15000) -> str:
        """Select most relevant sections for query within char budget."""
        if not sections:
            return ""
        
        total_size = sum(len(s) for s in sections)
        if total_size <= max_chars:
            return '\n\n'.join(sections)
        
        similar = self.find_similar(query, sections, top_k=len(sections), threshold=SEMANTIC_RELEVANCE_THRESHOLD)
        
        # Always include last 2 sections (most recent)
        recent_indices = set(range(max(0, len(sections) - 2), len(sections)))
        
        result = []
        used_indices = set()
        remaining = max_chars
        
        # Add recent first
        for idx in recent_indices:
            section = sections[idx]
            if len(section) <= remaining:
                result.append((idx, section))
                used_indices.add(idx)
                remaining -= len(section)
        
        # Add by relevance
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
# HIERARCHICAL MEMORY - Hot/Warm/Cold storage
# =============================================================================

class HierarchicalMemory:
    """
    Three-tier memory system:
    - Hot: Last 10 iterations, full detail
    - Warm: Next 40 iterations, summaries
    - Cold: Key points, permanent
    """
    
    def __init__(self):
        MEMORY_DIR.mkdir(exist_ok=True)
        self.hot_file = MEMORY_DIR / "hot.json"
        self.warm_file = MEMORY_DIR / "warm.json"
        self.cold_file = MEMORY_DIR / "cold.json"
        
        self.hot_limit = 10
        self.warm_limit = 40
        self.cold_limit = 200
    
    def _load(self, filepath: Path) -> List:
        if filepath.exists():
            try:
                return json.loads(filepath.read_text())
            except Exception:
                return []
        return []
    
    def _save(self, filepath: Path, data: List):
        try:
            filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            os.chmod(filepath, 0o666)
        except Exception as e:
            log_error(f"Failed to save {filepath}: {e}")
    
    def add_iteration(self, iteration: int, task_id: str, summary: str, 
                      details: str, key_points: List[str] = None):
        """Add iteration to memory."""
        entry = {
            'iteration': iteration,
            'task_id': task_id,
            'summary': summary,
            'details': details,
            'key_points': key_points or [],
            'timestamp': datetime.now().isoformat()
        }
        
        hot = self._load(self.hot_file)
        hot.append(entry)
        
        while len(hot) > self.hot_limit:
            old = hot.pop(0)
            self._add_to_warm(old)
        
        self._save(self.hot_file, hot)
    
    def _add_to_warm(self, entry: dict):
        warm = self._load(self.warm_file)
        warm_entry = {
            'iteration': entry['iteration'],
            'task_id': entry['task_id'],
            'summary': entry['summary'][:500],
            'key_points': entry.get('key_points', [])[:5],
            'timestamp': entry['timestamp']
        }
        warm.append(warm_entry)
        
        while len(warm) > self.warm_limit:
            old = warm.pop(0)
            self._add_to_cold(old)
        
        self._save(self.warm_file, warm)
    
    def _add_to_cold(self, entry: dict):
        cold = self._load(self.cold_file)
        for point in entry.get('key_points', []):
            if point and point not in cold:
                cold.append(point)
        cold = list(dict.fromkeys(cold))[-self.cold_limit:]
        self._save(self.cold_file, cold)
    
    def get_context(self, task_keywords: List[str] = None, max_chars: int = 30000) -> str:
        """Get memory context for prompt."""
        parts = []
        remaining = max_chars
        
        # Hot - full detail
        hot = self._load(self.hot_file)
        if hot:
            hot_text = "## Recent Iterations (detailed)\n"
            for entry in hot[-3:]:
                hot_text += f"\n### Iteration {entry['iteration']}: {entry['task_id']}\n"
                hot_text += entry.get('details', entry.get('summary', ''))[:1000]
                hot_text += "\n"
            parts.append(hot_text)
            remaining -= len(hot_text)
        
        # Warm - relevant summaries
        if remaining > 5000:
            warm = self._load(self.warm_file)
            if warm and task_keywords:
                scored = []
                for entry in warm:
                    text = entry.get('summary', '') + ' '.join(entry.get('key_points', []))
                    score = sum(1 for kw in task_keywords if kw.lower() in text.lower())
                    scored.append((score, entry))
                
                scored.sort(reverse=True, key=lambda x: x[0])
                relevant = [e for s, e in scored[:5] if s > 0]
                
                if relevant:
                    warm_text = "\n## Related Past Work\n"
                    for entry in relevant:
                        warm_text += f"- [{entry['task_id']}] {entry['summary'][:200]}\n"
                    parts.append(warm_text)
                    remaining -= len(warm_text)
        
        # Cold - key points
        if remaining > 2000:
            cold = self._load(self.cold_file)
            if cold:
                if task_keywords:
                    cold = [p for p in cold if any(kw.lower() in p.lower() for kw in task_keywords)]
                if cold:
                    cold_text = "\n## Key Learnings\n"
                    cold_text += '\n'.join(f"- {p}" for p in cold[-20:])
                    parts.append(cold_text)
        
        return '\n'.join(parts)
    
    def add_permanent(self, point: str):
        """Add point directly to cold storage."""
        cold = self._load(self.cold_file)
        point = point.strip()
        if point and point not in cold:
            cold.append(point)
            cold = cold[-self.cold_limit:]
            self._save(self.cold_file, cold)
    
    def compact_cold(self, semantic: 'SemanticSearch', threshold: float = SEMANTIC_COMPACT_THRESHOLD):
        """Compact cold storage by removing near-duplicates."""
        cold = self._load(self.cold_file)
        if len(cold) < 50:
            return  # Not enough to compact
        
        # Find unique entries
        unique = []
        for point in cold:
            if not unique:
                unique.append(point)
            elif not semantic.is_duplicate(point, unique, threshold=threshold):
                unique.append(point)
        
        if len(unique) < len(cold):
            log(f"Memory compaction: {len(cold)} -> {len(unique)} points")
            self._save(self.cold_file, unique)


# =============================================================================
# LEARNINGS MANAGER - Deduplication
# =============================================================================

class LearningsManager:
    """Manage learnings with deduplication."""
    
    def __init__(self, semantic: SemanticSearch):
        self.semantic = semantic
        self._entries = []
        self._load()
    
    def _load(self):
        if not LEARNINGS_FILE.exists():
            return
        content = LEARNINGS_FILE.read_text()
        sections = re.split(r'\n(?=###?\s)', content)
        self._entries = [s.strip() for s in sections if s.strip()]
    
    def add(self, learning: str) -> bool:
        """Add learning if not duplicate."""
        learning = learning.strip()
        if not learning:
            return False
        
        if self._entries and self.semantic.is_duplicate(learning, self._entries, threshold=SEMANTIC_DUPLICATE_THRESHOLD):
            return False
        
        self._entries.append(learning)
        self._save()
        return True
    
    def _save(self):
        content = '\n\n'.join(self._entries)
        try:
            LEARNINGS_FILE.write_text(content)
            os.chmod(LEARNINGS_FILE, 0o666)
        except Exception as e:
            log_error(f"Failed to save learnings: {e}")
    
    def get_relevant(self, query: str, max_chars: int = 15000) -> str:
        if not self._entries:
            return ""
        return self.semantic.select_relevant(query, self._entries, max_chars)
    
    def get_all(self) -> str:
        return '\n\n'.join(self._entries)
    
    def count(self) -> int:
        return len(self._entries)


# =============================================================================
# CONTEXT CONDENSER
# =============================================================================

class ContextCondenser:
    """Condense context with verification."""
    
    def __init__(self, semantic: SemanticSearch):
        self.semantic = semantic
        self.condense_history_file = RALPH_DIR / "condense_history.json"
        self.last_condense_iteration = 0
        self._load_state()
    
    def _load_state(self):
        if self.condense_history_file.exists():
            try:
                history = json.loads(self.condense_history_file.read_text())
                self.last_condense_iteration = history.get('last_iteration', 0)
            except Exception:
                pass
    
    def _save_state(self, iteration: int):
        history = {
            'last_iteration': iteration,
            'timestamp': datetime.now().isoformat()
        }
        try:
            self.condense_history_file.write_text(json.dumps(history, indent=2))
            os.chmod(self.condense_history_file, 0o666)
        except Exception:
            pass
        self.last_condense_iteration = iteration
    
    def should_condense(self, iteration: int, config: dict, 
                        context_size: int = 0, next_is_architect: bool = False) -> Tuple[bool, str]:
        """Check if condensation should run."""
        condense_interval = config.get('condenseInterval', 15)
        if condense_interval <= 0:
            return False, "disabled"
        
        iterations_since = iteration - self.last_condense_iteration
        
        if iterations_since >= condense_interval:
            return True, f"interval ({iterations_since} iterations)"
        
        if config.get('condenseBeforeArchitect', True) and next_is_architect and iterations_since >= 5:
            return True, "before architect"
        
        if context_size > CONTEXT_SIZE_LIMIT:
            return True, f"context size ({context_size} chars)"
        
        return False, ""
    
    def parse_condensed(self, output: str) -> Optional[str]:
        """Extract condensed context from output."""
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
    
    def extract_critical_facts(self, content: str) -> List[str]:
        """Extract critical facts that must be preserved."""
        facts = []
        
        # Errors
        for match in re.findall(r'(?:error|failed|exception|bug|fix)[:\s]+([^\n]{20,200})', content, re.I)[:10]:
            facts.append(f"ERROR: {match.strip()}")
        
        # Decisions
        for match in re.findall(r'(?:decided|chose|because)[:\s]+([^\n]{20,200})', content, re.I)[:10]:
            facts.append(f"DECISION: {match.strip()}")
        
        # Completions
        for match in re.findall(r'(?:completed|finished|done)[:\s]+([^\n]{10,150})', content, re.I)[:10]:
            facts.append(f"COMPLETED: {match.strip()}")
        
        return facts
    
    def verify_condensation(self, original: str, condensed: str) -> Tuple[bool, List[str]]:
        """Verify critical facts are preserved."""
        facts = self.extract_critical_facts(original)
        if not facts:
            return True, []
        
        missing = []
        preserved = 0
        condensed_lower = condensed.lower()
        
        for fact in facts:
            fact_content = fact.split(':', 1)[-1].strip().lower()
            key_words = [w for w in fact_content.split() if len(w) > 4][:5]
            
            if sum(1 for w in key_words if w in condensed_lower) >= len(key_words) * 0.5:
                preserved += 1
            elif self.semantic.compute_similarity(fact, condensed) > 0.3:
                preserved += 1
            else:
                missing.append(fact)
        
        is_valid = (preserved / len(facts)) >= 0.7 if facts else True
        return is_valid, missing
    
    def save_condensed(self, iteration: int, condensed: str, original: str = ""):
        """Save condensed context with verification."""
        # Verify if original provided
        if original:
            is_valid, missing = self.verify_condensation(original, condensed)
            if not is_valid and missing:
                condensed += "\n\n## Critical Facts (auto-preserved)\n"
                for fact in missing[:20]:
                    condensed += f"- {fact}\n"
                log(f"Added {len(missing)} missing facts to condensed")
        
        # Backup old
        if CONDENSED_FILE.exists():
            backup_dir = RALPH_DIR / "condense_backups"
            backup_dir.mkdir(exist_ok=True)
            backup_file = backup_dir / f"condensed_{self.last_condense_iteration}.md"
            try:
                shutil.copy2(CONDENSED_FILE, backup_file)
                # Keep only last 5 backups
                backups = sorted(backup_dir.glob("condensed_*.md"))
                for old in backups[:-5]:
                    old.unlink()
            except Exception:
                pass
        
        # Save new
        header = f"# Condensed Context (Iteration {iteration})\n"
        header += f"# Generated: {datetime.now().isoformat()}\n\n"
        try:
            CONDENSED_FILE.write_text(header + condensed)
            os.chmod(CONDENSED_FILE, 0o666)
            self._save_state(iteration)
            log(f"Saved condensed context: {len(condensed)} chars")
        except Exception as e:
            log_error(f"Failed to save condensed: {e}")
    
    def get_condensed(self) -> str:
        """Get current condensed context."""
        if CONDENSED_FILE.exists():
            return CONDENSED_FILE.read_text()
        return ""
    
    def get_recent_iterations_summary(self, num_iterations: int = 15) -> str:
        """Build summary of recent iterations."""
        if not ITERATIONS_DIR.exists():
            return ""
        
        summaries = []
        json_files = sorted(ITERATIONS_DIR.glob("iteration_*.json"))[-num_iterations:]
        
        for json_file in json_files:
            try:
                data = json.loads(json_file.read_text())
                iter_num = data.get('iteration', '?')
                iter_type = data.get('type', '?')
                result = data.get('result', 'unknown')
                summaries.append(f"### Iteration {iter_num} ({iter_type}): {result}")
            except Exception:
                continue
        
        return '\n\n'.join(summaries)


# =============================================================================
# METRICS - Track performance
# =============================================================================

class RalphMetrics:
    """Track performance metrics for monitoring and optimization."""
    
    def __init__(self):
        self.metrics_file = RALPH_DIR / "metrics.json"
        self._metrics = self._load()
    
    def _load(self) -> dict:
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
            "iteration_times": [],
            "context_sizes": [],
            "learnings_count": 0,
            "learnings_dedupe_skipped": 0,
            "started_at": None,
            "last_updated": None
        }
    
    def _save(self):
        self._metrics["last_updated"] = datetime.now().isoformat()
        try:
            self.metrics_file.write_text(json.dumps(self._metrics, indent=2))
            os.chmod(self.metrics_file, 0o666)
        except Exception:
            pass
    
    def record_iteration(self, success: bool, time_seconds: float, stuck: bool = False):
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
        self._metrics["iteration_times"].append(time_seconds)
        self._metrics["iteration_times"] = self._metrics["iteration_times"][-50:]
        self._save()
    
    def record_condense(self):
        self._metrics["condense_count"] += 1
        self._save()
    
    def record_learning(self, added: bool, duplicate: bool = False):
        if added:
            self._metrics["learnings_count"] += 1
        if duplicate:
            self._metrics["learnings_dedupe_skipped"] += 1
        self._save()
    
    def get_stats(self) -> dict:
        m = self._metrics
        total = m["total_iterations"]
        return {
            "total_iterations": total,
            "success_rate": m["successful_iterations"] / total if total > 0 else 0,
            "stuck_rate": m["stuck_count"] / total if total > 0 else 0,
            "avg_time": sum(m["iteration_times"]) / len(m["iteration_times"]) if m["iteration_times"] else 0,
            "condense_count": m["condense_count"],
            "learnings": m["learnings_count"],
            "dedupe_skipped": m["learnings_dedupe_skipped"]
        }


# =============================================================================
# EPOCHS - Milestone snapshots for long projects
# =============================================================================

class EpochManager:
    """Manage epoch snapshots for long-running projects."""
    
    def __init__(self):
        self.epochs_dir = RALPH_DIR / "epochs"
        self.epochs_dir.mkdir(exist_ok=True)
    
    def check_milestone(self, iteration: int) -> Optional[str]:
        """Check if milestone reached. Returns reason or None."""
        config = read_config()
        last_epoch = config.get("lastEpochIteration", 0)
        
        # Don't save too frequently
        if iteration - last_epoch < 25:
            return None
        
        prd = read_prd()
        stories = prd.get("userStories", [])
        done = sum(1 for s in stories if s.get("passes"))
        total = len(stories)
        
        # Check milestones
        if iteration > 0 and iteration % 100 == 0:
            return f"iteration_{iteration}"
        
        if total > 10:
            if done >= total * 0.5 and not config.get("50pctEpochSaved"):
                return "50pct_complete"
            if done >= total * 0.75 and not config.get("75pctEpochSaved"):
                return "75pct_complete"
        
        return None
    
    def save_epoch(self, iteration: int, reason: str) -> bool:
        """Save epoch snapshot."""
        config = read_config()
        current_epoch = config.get("currentEpoch", 0) + 1
        
        try:
            prd = read_prd()
            stories = prd.get("userStories", [])
            done = sum(1 for s in stories if s.get("passes"))
            total = len(stories)
            
            completed_tasks = [s.get("title", "?") for s in stories if s.get("passes")][-20:]
            
            epoch_data = {
                "epoch": current_epoch,
                "iteration": iteration,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "progress": {"done": done, "total": total},
                "phase": prd.get("phase", "unknown"),
                "verified": prd.get("verified", False),
                "recentTasks": completed_tasks
            }
            
            # Save epoch
            epoch_file = self.epochs_dir / f"epoch_{current_epoch:03d}.json"
            epoch_file.write_text(json.dumps(epoch_data, indent=2))
            os.chmod(epoch_file, 0o666)
            
            # Update config
            update_config("currentEpoch", current_epoch)
            update_config("lastEpochIteration", iteration)
            
            # Mark milestone flags
            if "50pct" in reason:
                update_config("50pctEpochSaved", True)
            if "75pct" in reason:
                update_config("75pctEpochSaved", True)
            
            log(f"Saved epoch {current_epoch}: {reason}")
            return True
            
        except Exception as e:
            log_error(f"Failed to save epoch: {e}")
            return False
    
    def get_epoch_context(self, max_epochs: int = 3) -> str:
        """Get context from recent epochs."""
        epoch_files = sorted(self.epochs_dir.glob("epoch_*.json"))[-max_epochs:]
        
        if not epoch_files:
            return ""
        
        parts = ["## Project Milestones"]
        for f in epoch_files:
            try:
                data = json.loads(f.read_text())
                parts.append(f"- Epoch {data['epoch']} (iter {data['iteration']}): {data['reason']}")
                parts.append(f"  Progress: {data['progress']['done']}/{data['progress']['total']}")
            except Exception:
                continue
        
        return '\n'.join(parts)


# =============================================================================
# STUCK DETECTOR - Detects when stuck and triggers recovery
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
    
    def __init__(self):
        self.stuck_file = RALPH_DIR / "stuck_history.json"
        self.max_task_attempts = 3
        self.max_same_errors = 5
        self.max_no_progress = 10
        self.max_verification_fails = 3
    
    def check_stuck(self, current_task_id: str, iteration: int) -> Tuple[bool, str]:
        """Check if we're stuck. Returns (is_stuck, reason)."""
        history = self._load_history()
        
        # Check task attempts
        task_attempts = [h for h in history if h.get('task_id') == current_task_id]
        if len(task_attempts) >= self.max_task_attempts:
            return True, f"Task {current_task_id} attempted {len(task_attempts)} times"
        
        # Check same error
        recent_errors = [h.get('error', '') for h in history[-self.max_same_errors:] if h.get('error')]
        if len(recent_errors) >= self.max_same_errors and len(set(recent_errors)) == 1:
            return True, f"Same error repeated {len(recent_errors)} times"
        
        # Check no progress
        recent = history[-self.max_no_progress:]
        if len(recent) >= self.max_no_progress:
            completed = sum(1 for h in recent if h.get('completed'))
            if completed == 0:
                return True, f"No task completed in last {self.max_no_progress} iterations"
        
        # Check verification failures
        recent_verifications = [h for h in history[-10:] if h.get('type') == 'verification']
        failed = [v for v in recent_verifications if not v.get('passed')]
        if len(failed) >= self.max_verification_fails:
            return True, f"Verification failed {len(failed)} times in a row"
        
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
        history = history[-100:]  # Keep last 100
        self._save_history(history)
    
    def get_recovery_strategy(self, reason: str) -> str:
        """
        Determine recovery strategy:
        - skip_task: Skip current task, mark as blocked
        - rollback: Rollback recent changes
        - escalate: Add architect review task
        """
        if 'attempted' in reason:
            return 'skip_task'
        elif 'error repeated' in reason:
            return 'rollback'
        elif 'No task completed' in reason:
            return 'escalate'
        elif 'Verification failed' in reason:
            return 'rollback'
        return 'skip_task'
    
    def get_task_attempts(self, task_id: str) -> str:
        """Get formatted history of previous attempts."""
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
        """Clear history after successful recovery."""
        self._save_history([])
    
    def _load_history(self) -> List[dict]:
        if self.stuck_file.exists():
            try:
                return json.loads(self.stuck_file.read_text())
            except Exception:
                return []
        return []
    
    def _save_history(self, history: List[dict]):
        try:
            self.stuck_file.write_text(json.dumps(history, indent=2, ensure_ascii=False))
            os.chmod(self.stuck_file, 0o666)
        except Exception:
            pass


# =============================================================================
# CIRCUIT BREAKER - Resilience for external services
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for external services.
    
    States:
    - CLOSED: normal operation
    - OPEN: failing, reject requests
    - HALF_OPEN: testing if recovered
    """
    
    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_timeout: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = 0.0
    
    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                log(f"Circuit {self.name}: OPEN -> HALF_OPEN")
                return True
            return False
        
        return True  # HALF_OPEN allows
    
    def record_success(self):
        """Record successful call."""
        if self.state == "HALF_OPEN":
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
            log(f"Circuit {self.name}: HALF_OPEN -> OPEN (still failing)")
        elif self.failures >= self.failure_threshold:
            self.state = "OPEN"
            log(f"Circuit {self.name}: CLOSED -> OPEN (threshold reached)")


# =============================================================================
# DISK SPACE MONITOR - Cleanup when low space
# =============================================================================

MIN_FREE_SPACE_MB = 500  # Minimum free space before cleanup

class DiskSpaceMonitor:
    """Monitors disk space and triggers cleanup."""
    
    def get_free_space_mb(self) -> float:
        """Get free disk space in MB."""
        try:
            stat = os.statvfs(RALPH_DIR)
            free_bytes = stat.f_bavail * stat.f_frsize
            return free_bytes / (1024 * 1024)
        except Exception:
            return float('inf')
    
    def is_low_space(self) -> bool:
        return self.get_free_space_mb() < MIN_FREE_SPACE_MB
    
    def rotate_logs(self, emergency: bool = False) -> dict:
        """Rotate logs to free space."""
        import gzip
        stats = {"truncated": 0, "deleted": 0, "compressed": 0}
        
        # 1. Truncate daemon log
        max_size = 500_000 if emergency else 1_000_000
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > max_size:
            try:
                content = LOG_FILE.read_text()
                LOG_FILE.write_text(f"...(rotated {datetime.now().isoformat()})...\n" + content[-max_size:])
                os.chmod(LOG_FILE, 0o666)
                stats["truncated"] += 1
            except Exception:
                pass
        
        # 2. Delete old condense backups
        keep_count = 3 if emergency else 5
        backup_dir = RALPH_DIR / "condense_backups"
        if backup_dir.exists():
            backups = sorted(backup_dir.glob("condensed_*.md"))
            for old in backups[:-keep_count]:
                try:
                    old.unlink()
                    stats["deleted"] += 1
                except Exception:
                    pass
        
        # 3. Compress old iteration logs (>7 days for normal, >3 days for emergency)
        if ITERATIONS_DIR.exists():
            days = 3 if emergency else 7
            cutoff = time.time() - (days * 24 * 3600)
            for log_file in ITERATIONS_DIR.glob("*.log"):
                try:
                    if log_file.stat().st_mtime < cutoff:
                        gz_path = log_file.with_suffix(".log.gz")
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(gz_path, 'wb') as f_out:
                                f_out.writelines(f_in)
                        log_file.unlink()
                        stats["compressed"] += 1
                except Exception:
                    pass
        
        # 4. Limit iteration JSON files
        keep_json = 100 if emergency else 500
        if ITERATIONS_DIR.exists():
            json_files = sorted(ITERATIONS_DIR.glob("*.json"))
            for old in json_files[:-keep_json]:
                try:
                    old.unlink()
                    stats["deleted"] += 1
                except Exception:
                    pass
        
        # 5. Trim semantic cache (keep recent entries only)
        cache_file = RALPH_DIR / "semantic_cache.json"
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
                keep_cache = 3000 if emergency else 8000
                if len(cache) > keep_cache:
                    keys = list(cache.keys())[-keep_cache:]
                    cache = {k: cache[k] for k in keys}
                    cache_file.write_text(json.dumps(cache))
                    os.chmod(cache_file, 0o666)
                    stats["truncated"] += 1
            except Exception:
                pass
        
        # 6. Delete old compressed logs in emergency
        if emergency and ITERATIONS_DIR.exists():
            gz_files = sorted(ITERATIONS_DIR.glob("*.log.gz"))
            for old in gz_files[:-50]:
                try:
                    old.unlink()
                    stats["deleted"] += 1
                except Exception:
                    pass
        
        if stats["truncated"] or stats["deleted"] or stats["compressed"]:
            log(f"Disk cleanup: truncated={stats['truncated']}, deleted={stats['deleted']}, compressed={stats['compressed']}")
        
        return stats


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

semantic_search = None
memory = None
learnings_mgr = None
condenser = None
metrics = None
epochs = None
stuck_detector = None
circuit_breaker = None
disk_monitor = None


def init_managers():
    """Initialize global managers."""
    global semantic_search, memory, learnings_mgr, condenser, metrics, epochs
    global stuck_detector, circuit_breaker, disk_monitor
    
    semantic_search = SemanticSearch()
    memory = HierarchicalMemory()
    learnings_mgr = LearningsManager(semantic_search)
    condenser = ContextCondenser(semantic_search)
    metrics = RalphMetrics()
    epochs = EpochManager()
    stuck_detector = StuckDetector()
    circuit_breaker = CircuitBreaker("openhands", failure_threshold=5, recovery_timeout=120)
    disk_monitor = DiskSpaceMonitor()
    
    log("All managers initialized")


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================

def get_current_task(prd: dict) -> Optional[dict]:
    """Get next pending task from PRD (skip blocked tasks)."""
    stories = prd.get("userStories", [])
    for story in stories:
        if not story.get("passes", False) and not story.get("blocked", False):
            return story
    return None


def mark_task_done(task_id: str, passed: bool = True):
    """Mark task as done in PRD."""
    prd = read_prd()
    for story in prd.get("userStories", []):
        if story.get("id") == task_id:
            story["passes"] = passed
            story["completedAt"] = datetime.now().isoformat()
            break
    save_prd(prd)
    log(f"Task {task_id} marked as {'PASS' if passed else 'FAIL'}")


def set_pending_verification(task_id: str, iteration: int, task_title: str = "", task_desc: str = ""):
    """Set task for verification with full context."""
    config = read_config()
    config["pendingTaskVerification"] = {
        "taskId": task_id,
        "taskTitle": task_title,
        "taskDescription": task_desc,
        "claimedAt": datetime.now().isoformat(),
        "claimedIteration": iteration
    }
    save_config(config)


def clear_pending_verification():
    """Clear pending verification."""
    config = read_config()
    config["pendingTaskVerification"] = None
    save_config(config)


def should_run_condense(config: dict, iteration: int) -> Tuple[bool, str]:
    """Check if context condensation should run using condenser."""
    if condenser is None:
        return False, "condenser not initialized"
    
    # Calculate context size
    context_size = 0
    if LEARNINGS_FILE.exists():
        context_size += len(LEARNINGS_FILE.read_text())
    if CONDENSED_FILE.exists():
        context_size += len(CONDENSED_FILE.read_text())
    
    # Check if next is architect
    architect_interval = config.get("architectInterval", 10)
    next_is_architect = (architect_interval > 0 and 
                         (iteration + 1) % architect_interval == 0)
    
    return condenser.should_condense(iteration, config, context_size, next_is_architect)


def get_iteration_type(config: dict) -> str:
    """Determine iteration type based on state."""
    prd = read_prd()
    iteration = config.get("currentIteration", 0)
    
    # Check for pending verification first
    pending = config.get("pendingTaskVerification")
    if pending and pending.get("taskId"):
        return "task_verify"
    
    phase = prd.get("phase", "planning")
    
    # Planning phase
    if phase == "planning":
        # Check if PLAN task is done
        for story in prd.get("userStories", []):
            if story.get("id") == "PLAN" and story.get("passes"):
                # Planning done, move to execution
                prd["phase"] = "execution"
                save_prd(prd)
                return "worker"
        return "planning"
    
    # Verification phase (final project verification)
    if phase == "verification":
        return "verification"
    
    # Execution phase - check for condense/architect
    
    # Check for condense
    should_condense, reason = should_run_condense(config, iteration)
    if should_condense:
        log(f"Condense needed: {reason}")
        return "condense"
    
    # Check for architect interval
    architect_interval = config.get("architectInterval", 10)
    if architect_interval > 0 and iteration > 0 and iteration % architect_interval == 0:
        return "architect"
    
    # Check if all tasks done -> verification phase
    current_task = get_current_task(prd)
    if not current_task:
        prd["phase"] = "verification"
        save_prd(prd)
        return "verification"
    
    return "worker"


def parse_ralph_tags(output: str) -> dict:
    """Parse ralph tags from iteration output."""
    result = {
        "task_done": None,
        "task_verified": None,
        "stuck": False,
        "project_verified": False,
        "learnings": []
    }
    
    # TASK_DONE:TASK-XXX
    match = re.search(r'<ralph>TASK_DONE:([^<]+)</ralph>', output)
    if match:
        result["task_done"] = match.group(1).strip()
    
    # TASK_VERIFIED:TASK-XXX:PASS or TASK_VERIFIED:TASK-XXX:FAIL:reason
    match = re.search(r'<ralph>TASK_VERIFIED:([^:]+):(\w+)(?::([^<]+))?</ralph>', output)
    if match:
        result["task_verified"] = {
            "task_id": match.group(1).strip(),
            "passed": match.group(2).upper() == "PASS",
            "reason": match.group(3).strip() if match.group(3) else None
        }
    
    # STUCK
    if '<ralph>STUCK</ralph>' in output:
        result["stuck"] = True
    
    # PROJECT_VERIFIED
    if '<ralph>PROJECT_VERIFIED</ralph>' in output:
        result["project_verified"] = True
    
    # LEARNING:text
    for match in re.finditer(r'<ralph>LEARNING:([^<]+)</ralph>', output):
        result["learnings"].append(match.group(1).strip())
    
    return result


def append_learnings(learnings: List[str], iteration: int):
    """Append learnings with deduplication via semantic search."""
    if not learnings or not learnings_mgr:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    added_count = 0
    
    for learning in learnings:
        learning = learning.strip()
        if not learning:
            continue
        
        # Format learning with context
        formatted = f"## Iteration {iteration} ({timestamp})\n- {learning}"
        
        if learnings_mgr.add(formatted):
            added_count += 1
            # Also add key point to permanent memory
            if memory:
                memory.add_permanent(learning)
            if metrics:
                metrics.record_learning(added=True)
        else:
            if metrics:
                metrics.record_learning(added=False, duplicate=True)
    
    if added_count > 0:
        log(f"Added {added_count}/{len(learnings)} new learnings")


def read_file_limited(filepath: Path, max_chars: int = 50000) -> str:
    """Read file with character limit."""
    if not filepath.exists():
        return ""
    try:
        content = filepath.read_text()
        if len(content) > max_chars:
            return f"...(truncated, showing last {max_chars} chars)...\n" + content[-max_chars:]
        return content
    except Exception:
        return ""


def get_last_iteration_summary() -> str:
    """Get summary of last iteration for context."""
    try:
        # Find latest iteration log
        logs = sorted(ITERATIONS_DIR.glob("iteration_*.log"), reverse=True)
        if not logs:
            return "(First iteration)"
        
        last_log = logs[0]
        content = last_log.read_text()
        
        # Get last 2000 chars
        if len(content) > 2000:
            content = "...\n" + content[-2000:]
        
        return content
    except Exception:
        return "(Could not read last iteration)"


def get_completed_tasks_summary(prd: dict) -> str:
    """Get summary of completed tasks."""
    stories = prd.get("userStories", [])
    completed = [s for s in stories if s.get("passes")]
    
    if not completed:
        return "(No tasks completed yet)"
    
    lines = []
    for s in completed[-10:]:  # Last 10 completed
        lines.append(f"  [DONE] [{s.get('id', '?')}] {s.get('title', '?')}")
    
    if len(completed) > 10:
        lines.insert(0, f"  ... and {len(completed) - 10} more completed tasks")
    
    return "\n".join(lines)


def get_project_summary() -> str:
    """Get current project state summary."""
    prd = read_prd()
    stories = prd.get("userStories", [])
    done = sum(1 for s in stories if s.get("passes"))
    total = len(stories)
    phase = prd.get("phase", "unknown")
    
    return f"Phase: {phase}, Progress: {done}/{total} tasks complete"


def substitute_prompt_vars(template: str, config: dict, iter_type: str) -> str:
    """Substitute variables in prompt template with smart context selection."""
    prd = read_prd()
    task = get_current_task(prd)
    stories = prd.get("userStories", [])
    done = sum(1 for s in stories if s.get("passes"))
    total = len(stories)
    iteration = config.get("currentIteration", 0)
    
    # For verification - use pending task info
    pending = config.get("pendingTaskVerification", {}) or {}
    
    # Task info - use pending info for task_verify, current task otherwise
    if iter_type == "task_verify" and pending:
        task_id = pending.get("taskId", "UNKNOWN")
        task_title = pending.get("taskTitle", "")
        task_desc = pending.get("taskDescription", "")
        verify_task_id = task_id
    else:
        task_id = task.get("id", "UNKNOWN") if task else "NONE"
        task_title = task.get("title", "") if task else ""
        task_desc = task.get("description", "") if task else ""
        verify_task_id = task_id
    
    # Read base context files
    mission = read_file_limited(MISSION_FILE, MISSION_LIMIT)
    architecture = read_file_limited(ARCHITECTURE_FILE, ARCHITECTURE_LIMIT)
    
    # Smart learnings selection using managers
    if condenser and condenser.get_condensed():
        # Use condensed + recent learnings
        learnings = condenser.get_condensed()
        if learnings_mgr and task:
            # Add task-relevant recent learnings
            query = f"{task_title} {task_desc}"
            recent = learnings_mgr.get_relevant(query, max_chars=5000)
            if recent:
                learnings += "\n\n## Recent Relevant Learnings\n" + recent
    elif learnings_mgr and task:
        # Use semantic search for relevant learnings
        query = f"{task_title} {task_desc}"
        learnings = learnings_mgr.get_relevant(query, max_chars=LEARNINGS_LIMIT)
    else:
        learnings = read_file_limited(LEARNINGS_FILE, LEARNINGS_LIMIT)
    
    # Get hierarchical memory context
    memory_context = ""
    if memory and task:
        keywords = [w for w in f"{task_title} {task_desc}".split() if len(w) > 3]
        memory_context = memory.get_context(keywords, max_chars=MEMORY_CONTEXT_LIMIT)
    
    # Build sections
    last_iteration = get_last_iteration_summary()
    completed_tasks = get_completed_tasks_summary(prd)
    project_summary = get_project_summary()
    
    # Guardrails
    guardrails_file = RALPH_DIR / "guardrails.md"
    guardrails_section = read_file_limited(guardrails_file, 5000)
    
    # AGENTS.md for build instructions
    agents_file = RALPH_DIR / "AGENTS.md"
    agents_section = read_file_limited(agents_file, 5000)
    
    # Previous attempts on this task (from stuck detector)
    previous_attempts = ""
    if stuck_detector and task_id:
        previous_attempts = stuck_detector.get_task_attempts(task_id)
    
    fix_history = ""
    
    # Get epoch context for long projects
    epoch_context = ""
    if epochs:
        epoch_context = epochs.get_epoch_context()
    
    # Variable substitutions
    substitutions = {
        "${iteration}": str(iteration),
        "${task_id}": task_id,
        "${task_title}": task_title,
        "${task_description}": task_desc,
        "${done_tasks}": str(done),
        "${total_tasks}": str(total),
        "${mission}": mission,
        "${learnings}": learnings,
        "${architecture}": architecture,
        "${last_iteration}": last_iteration,
        "${completed_tasks}": completed_tasks,
        "${project_summary}": project_summary,
        "${guardrails_section}": guardrails_section,
        "${agents_section}": agents_section,
        "${previous_attempts}": previous_attempts,
        "${fix_history}": fix_history,
        "${verify_task_id}": verify_task_id,
        "${memory_context}": memory_context,
        "${epoch_context}": epoch_context,
    }
    
    result = template
    for var, value in substitutions.items():
        result = result.replace(var, value)
    
    return result


def load_prompt_template(iter_type: str) -> str:
    """Load prompt template for iteration type.
    
    Priority:
    1. /workspace/.ralph/prompts/{iter_type}.md (project-specific)
    2. /workspace/.ralph/prompts/global/{iter_type}.md (global templates synced from host)
    3. Fallback built-in prompts
    
    Prompts are read fresh each iteration - edit on host, applies next iteration.
    """
    # Project-specific prompts
    project_prompt = RALPH_DIR / "prompts" / f"{iter_type}.md"
    if project_prompt.exists():
        return project_prompt.read_text()
    
    # Global templates (synced from host templates/ralph/)
    global_prompt = RALPH_DIR / "prompts" / "global" / f"{iter_type}.md"
    if global_prompt.exists():
        return global_prompt.read_text()
    
    # Minimal fallback prompts
    fallback_prompts = {
        "planning": """# Ralph Planning - Iteration ${iteration}

Read .ralph/MISSION.md and create execution plan.
Break down into small tasks in .ralph/prd.json.

When done: <ralph>TASK_DONE:PLAN</ralph>""",
        
        "worker": """# Ralph Worker - Iteration ${iteration}
Task: ${task_id} - ${task_title}

${mission}

Implement this task. Commit your changes.
When done: <ralph>TASK_DONE:${task_id}</ralph>

If stuck: <ralph>STUCK</ralph>""",
        
        "task_verify": """# Ralph Independent Verification - Iteration ${iteration}

## Task Being Verified
ID: ${verify_task_id}
Title: ${task_title}
Description: ${task_description}

## Instructions
Worker claimed this task is done. You must INDEPENDENTLY verify:
1. Check actual code changes match the task requirements
2. Run relevant tests
3. Verify the implementation is correct and complete

## Output
If PASS (task is truly complete): <ralph>TASK_VERIFIED:${verify_task_id}:PASS</ralph>
If FAIL (not done or buggy): <ralph>TASK_VERIFIED:${verify_task_id}:FAIL:specific reason</ralph>""",
        
        "architect": """# Ralph Architect Review - Iteration ${iteration}

Review architecture and progress. Update ARCHITECTURE.md.
Progress: ${done_tasks}/${total_tasks}

${architecture}""",
        
        "verification": """# Project Verification - Iteration ${iteration}

Verify entire project is complete. Run full tests.

If complete: <ralph>PROJECT_VERIFIED</ralph>
If not: <ralph>PROJECT_NOT_VERIFIED:reason</ralph>""",
        
        "condense": """# Context Condensation - Iteration ${iteration}

Summarize progress and learnings. Update LEARNINGS.md with key facts only.
Remove redundant information, keep essential knowledge."""
    }
    
    return fallback_prompts.get(iter_type, f"Continue working. Iteration type: {iter_type}")


def build_prompt(config: dict, iter_type: str) -> str:
    """Build full prompt for iteration with variable substitution."""
    template = load_prompt_template(iter_type)
    return substitute_prompt_vars(template, config, iter_type)


def handle_iteration_result(iteration: int, iter_type: str, output: str, config: dict) -> str:
    """Handle iteration result and update state. Returns result status."""
    tags = parse_ralph_tags(output)
    result = "unknown"
    
    # Save learnings
    if tags["learnings"]:
        append_learnings(tags["learnings"], iteration)
    
    # Handle based on iteration type
    if iter_type == "task_verify":
        if tags["task_verified"]:
            tv = tags["task_verified"]
            if tv["passed"]:
                # Verification passed - mark task done
                mark_task_done(tv["task_id"], True)
                clear_pending_verification()
                result = "verified_pass"
                log(f"Task {tv['task_id']} verified PASS")
            else:
                # Verification failed - task needs more work
                clear_pending_verification()
                result = "verified_fail"
                log(f"Task {tv['task_id']} verified FAIL: {tv.get('reason', 'unknown')}")
        else:
            # No verification tag - count retry attempts
            pending = config.get("pendingTaskVerification", {}) or {}
            task_id = pending.get("taskId")
            retry_count = pending.get("retryCount", 0) + 1
            
            if retry_count >= MAX_VERIFY_RETRIES:
                # After max retries, assume fail and let worker retry
                log_warning(f"Task {task_id} - {MAX_VERIFY_RETRIES} verification attempts without tag, treating as FAIL")
                clear_pending_verification()
                result = "verified_fail"
            else:
                # Keep pending, increment retry counter - will verify again
                config = read_config()
                if config.get("pendingTaskVerification"):
                    config["pendingTaskVerification"]["retryCount"] = retry_count
                    save_config(config)
                result = "verify_retry"
                log_warning(f"Task {task_id} - no verify tag, retry {retry_count}/{MAX_VERIFY_RETRIES}")
    
    elif iter_type == "worker":
        if tags["task_done"]:
            task_id = tags["task_done"]
            # Get task info for verification context
            prd = read_prd()
            task_info = None
            for s in prd.get("userStories", []):
                if s.get("id") == task_id:
                    task_info = s
                    break
            
            # Set for verification if enabled
            require_verify = config.get("requireVerification", True)
            if require_verify:
                set_pending_verification(
                    task_id, iteration,
                    task_title=task_info.get("title", "") if task_info else "",
                    task_desc=task_info.get("description", "") if task_info else ""
                )
                result = "pending_verify"
                log(f"Task {task_id} claimed done, pending verification")
            else:
                mark_task_done(task_id, True)
                result = "task_done"
                log(f"Task {task_id} done (no verification)")
        elif tags["stuck"]:
            result = "stuck"
            log_warning("Worker reported STUCK")
            # Increment stuck counter
            stuck_count = config.get("stuckCount", 0) + 1
            update_config("stuckCount", stuck_count)
        else:
            result = "working"
    
    elif iter_type == "planning":
        if tags["task_done"] and "PLAN" in tags["task_done"].upper():
            mark_task_done("PLAN", True)
            result = "planning_done"
            log("Planning phase complete")
        else:
            result = "planning"
    
    elif iter_type == "verification":
        if tags["project_verified"]:
            prd = read_prd()
            prd["verified"] = True
            prd["verifiedAt"] = datetime.now().isoformat()
            save_prd(prd)
            update_config("status", "complete")
            result = "project_verified"
            log("PROJECT VERIFIED - All done!")
        else:
            result = "verification_pending"
    
    elif iter_type == "architect":
        result = "architect_done"
    
    elif iter_type == "condense":
        # Use condenser to parse and save with verification
        if condenser:
            condensed_text = condenser.parse_condensed(output)
            if condensed_text:
                # Get original content for verification
                original = ""
                if LEARNINGS_FILE.exists():
                    original = LEARNINGS_FILE.read_text()
                original += "\n" + condenser.get_recent_iterations_summary()
                
                condenser.save_condensed(iteration, condensed_text, original)
                result = "condense_done"
            else:
                result = "condense_failed"
                log_warning("Failed to parse condensed output")
        else:
            # Fallback to simple save
            match = re.search(r'```condensed\s*\n(.*?)\n```', output, re.DOTALL)
            if match:
                condensed = match.group(1).strip()
                try:
                    header = f"# Condensed Context (Iteration {iteration})\n"
                    header += f"# Generated: {datetime.now().isoformat()}\n\n"
                    CONDENSED_FILE.write_text(header + condensed)
                    os.chmod(CONDENSED_FILE, 0o666)
                    log(f"Saved condensed context: {len(condensed)} chars")
                except Exception as e:
                    log_error(f"Failed to save condensed: {e}")
            result = "condense_done"
    
    return result


def run_iteration(config: dict) -> Tuple[bool, str]:
    """Run single iteration. Returns (success, result_type)."""
    global current_process
    
    iteration = config.get("currentIteration", 0) + 1
    update_config("currentIteration", iteration)
    
    # Check circuit breaker
    if circuit_breaker and not circuit_breaker.can_proceed():
        log_warning("Circuit breaker OPEN - waiting for recovery")
        time.sleep(30)
        return False, "circuit_open"
    
    iter_type = get_iteration_type(config)
    log(f"=== Iteration {iteration} ({iter_type}) ===")
    add_progress(iteration, f"Starting {iter_type} iteration")
    
    # Check if stuck on current task
    prd = read_prd()
    current_task = get_current_task(prd)
    if stuck_detector and current_task and iter_type == "worker":
        task_id = current_task.get("id", "")
        is_stuck, reason = stuck_detector.check_stuck(task_id, iteration)
        if is_stuck:
            log_warning(f"STUCK detected: {reason}")
            strategy = stuck_detector.get_recovery_strategy(reason)
            
            if strategy == "skip_task":
                # Mark task as blocked and move on
                current_task["blocked"] = True
                current_task["blockedReason"] = reason
                save_prd(prd)
                stuck_detector.clear_history()
                return True, "task_skipped"
            elif strategy == "escalate":
                # Force architect review
                iter_type = "architect"
                log("Escalating to architect review")
    
    # Build prompt
    prompt = build_prompt(config, iter_type)
    
    # Create output file
    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = ITERATIONS_DIR / f"iteration_{iteration:04d}.log"
    
    # Build command
    escaped_prompt = shlex.quote(prompt)
    cmd = f"openhands --headless --json -t {escaped_prompt}"
    
    start_time = time.time()
    exit_code = -1
    
    try:
        with open(output_file, "w") as f:
            current_process = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd="/workspace"
            )
            
            # Wait with timeout
            timeout = config.get("sessionTimeoutSeconds", 1800)
            try:
                exit_code = current_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log_warning(f"Iteration {iteration} timed out after {timeout}s")
                current_process.terminate()
                try:
                    current_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current_process.kill()
                    current_process.wait()
                return False, "timeout"
        
        current_process = None
        elapsed = time.time() - start_time
        
        # Read and parse output
        output = ""
        if output_file.exists():
            output = output_file.read_text()
        
        # Handle result
        result = handle_iteration_result(iteration, iter_type, output, config)
        
        # Save iteration metadata
        meta_file = ITERATIONS_DIR / f"iteration_{iteration:04d}.json"
        meta_file.write_text(json.dumps({
            "iteration": iteration,
            "type": iter_type,
            "result": result,
            "exit_code": exit_code,
            "duration_seconds": int(elapsed),
            "timestamp": datetime.now().isoformat(),
            "output_length": len(output)
        }, indent=2))
        try:
            os.chmod(meta_file, 0o666)
        except Exception:
            pass
        
        # Add to hierarchical memory
        if memory:
            prd = read_prd()
            task = get_current_task(prd)
            task_id = task.get("id", "unknown") if task else iter_type
            
            # Extract key points from output
            key_points = []
            tags = parse_ralph_tags(output)
            if tags["learnings"]:
                key_points.extend(tags["learnings"][:5])
            if tags["task_done"]:
                key_points.append(f"Completed: {tags['task_done']}")
            
            memory.add_iteration(
                iteration=iteration,
                task_id=task_id,
                summary=f"{iter_type}: {result}",
                details=output[-2000:] if len(output) > 2000 else output,
                key_points=key_points
            )
        
        # Record metrics
        if metrics:
            is_stuck = result == "stuck"
            metrics.record_iteration(success=True, time_seconds=elapsed, stuck=is_stuck)
            if result == "condense_done":
                metrics.record_condense()
        
        # Record in stuck detector
        if stuck_detector and current_task:
            task_id = current_task.get("id", "")
            completed = result in ["task_done", "pending_verify", "verified_pass"]
            error_msg = None
            if result == "stuck":
                error_msg = "Worker reported STUCK"
            elif result == "verified_fail":
                error_msg = "Verification failed"
            stuck_detector.record_attempt(task_id, iteration, completed, error_msg, iter_type)
        
        # Record circuit breaker success
        if circuit_breaker:
            circuit_breaker.record_success()
        
        # Check for epoch milestone
        if epochs:
            milestone = epochs.check_milestone(iteration)
            if milestone:
                epochs.save_epoch(iteration, milestone)
        
        add_progress(iteration, f"Completed: {result} ({int(elapsed)}s)")
        log(f"Iteration {iteration} complete: {result} ({int(elapsed)}s)")
        
        return True, result
        
    except Exception as e:
        log_error(f"Iteration {iteration} failed: {e}")
        log_error(traceback.format_exc())
        current_process = None
        
        # Record failed iteration
        if metrics:
            metrics.record_iteration(success=False, time_seconds=0)
        
        # Record circuit breaker failure
        if circuit_breaker:
            circuit_breaker.record_failure(str(e))
        
        return False, "error"


def ensure_open_permissions():
    """Set open permissions on ralph directories so host can access without sudo."""
    dirs = [RALPH_DIR, ITERATIONS_DIR, MEMORY_DIR, RALPH_DIR / "condense_backups", RALPH_DIR / "epochs"]
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o777)
        except Exception:
            pass


def main():
    """Main daemon loop with retry logic and error recovery."""
    global shutdown_requested
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Ensure directories exist with open permissions
    ensure_open_permissions()
    
    # Write PID file with open permissions
    PID_FILE.write_text(str(os.getpid()))
    try:
        os.chmod(PID_FILE, 0o666)
    except Exception:
        pass
    
    log("=" * 50)
    log("Ralph Daemon Started (Full-featured)")
    log(f"PID: {os.getpid()}")
    log(f"Working directory: /workspace")
    log("=" * 50)
    
    # Initialize managers
    init_managers()
    
    consecutive_errors = 0
    retry_delay = BASE_DELAY
    
    try:
        while not shutdown_requested:
            write_heartbeat()
            
            config = read_config()
            status = config.get("status", "paused")
            
            # Auto-promote "starting" to "running" now that daemon is alive
            if status == "starting":
                log("Daemon alive - promoting status to 'running'")
                update_config("status", "running")
                status = "running"
            
            if status == "running":
                # Check disk space
                if disk_monitor and disk_monitor.is_low_space():
                    log_warning("Low disk space - running cleanup")
                    disk_monitor.rotate_logs(emergency=True)
                
                # Periodic maintenance (every 50 iterations)
                current_iter = config.get("currentIteration", 0)
                if current_iter > 0 and current_iter % 50 == 0:
                    # Log rotation
                    if disk_monitor:
                        disk_monitor.rotate_logs(emergency=False)
                    
                    # Memory compaction every 100 iterations
                    if current_iter % 100 == 0 and memory and semantic_search:
                        memory.compact_cold(semantic_search)
                
                # Check max iterations
                max_iter = config.get("maxIterations", 0)
                
                if max_iter > 0 and current_iter >= max_iter:
                    log(f"Reached max iterations ({max_iter})")
                    update_config("status", "complete")
                    add_progress(current_iter, "Max iterations reached")
                    continue
                
                # Check if project already complete
                prd = read_prd()
                if prd.get("verified"):
                    log("Project already verified complete")
                    update_config("status", "complete")
                    continue
                
                # Check stuck counter
                stuck_count = config.get("stuckCount", 0)
                if stuck_count >= 3:
                    log_warning(f"Stuck {stuck_count} times, requesting human help")
                    update_config("status", "needs_help")
                    add_progress(current_iter, "Stuck - needs human intervention")
                    continue
                
                # Run iteration
                success, result = run_iteration(config)
                
                if success:
                    consecutive_errors = 0
                    retry_delay = BASE_DELAY
                    
                    # Reset stuck counter on progress
                    if result not in ["stuck", "working", "unknown"]:
                        update_config("stuckCount", 0)
                else:
                    consecutive_errors += 1
                    log_error(f"Iteration failed (consecutive: {consecutive_errors})")
                    
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        log_error(f"Too many errors ({consecutive_errors}), pausing")
                        update_config("status", "error")
                        add_progress(current_iter, f"Paused due to {consecutive_errors} consecutive errors")
                        consecutive_errors = 0
                    else:
                        # Exponential backoff
                        log(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 300)  # Max 5 min
                        continue
                
                # Pause between iterations
                pause = config.get("pauseBetweenSeconds", 10)
                if pause > 0:
                    time.sleep(pause)
                
            elif status == "paused":
                # Check periodically for resume
                time.sleep(5)
                
            elif status == "needs_help":
                # Wait for human intervention
                time.sleep(30)
                
            elif status in ["stopped", "complete"]:
                # Final states - keep daemon alive but idle
                time.sleep(10)
                
            elif status == "error":
                # Error state - wait for manual intervention
                time.sleep(30)
                
            else:
                # Unknown status, treat as paused
                log_warning(f"Unknown status: {status}, treating as paused")
                time.sleep(5)
    
    except KeyboardInterrupt:
        log("Keyboard interrupt received")
    except Exception as e:
        log_error(f"Fatal daemon error: {e}")
        log_error(traceback.format_exc())
    finally:
        # Cleanup
        log("Ralph daemon shutting down")
        try:
            PID_FILE.unlink()
        except Exception:
            pass
        log("Goodbye!")


if __name__ == "__main__":
    main()
