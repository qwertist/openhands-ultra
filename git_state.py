#!/usr/bin/env python3
"""
Git-Native State Management for Ralph v5.0

All state stored in git instead of JSON files:
- State: git refs (refs/ralph/*)
- Checkpoints: git tags (ralph/cp/*)
- Iterations: commit messages with [Ralph:Iter:N]
- Learnings: git notes (refs/notes/learnings)
- Handoffs: git notes (refs/notes/handoff)
- Context: git notes (refs/notes/context)

Benefits:
- Survives crashes (state is in git, not memory)
- Full history via git log
- Easy rollback via git reset/checkout
- Atomic operations
- Works with any git tooling
"""

import json
import os
import secrets
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger('ralph.git_state')

# =============================================================================
# TASK ID GENERATION (Bead-style)
# =============================================================================

def generate_task_id(prefix: str = "oh") -> str:
    """
    Generate bead-style task ID: prefix-xxxxx
    
    Examples: oh-k7m2x, oh-a3b1c, oh-p9n4q
    
    Args:
        prefix: 2-10 alphanumeric characters
    
    Raises:
        ValueError: if prefix is invalid
    """
    # Validate prefix
    if not prefix or not prefix.isalnum() or len(prefix) < 2 or len(prefix) > 10:
        raise ValueError(f"Invalid prefix: {prefix!r} (must be 2-10 alphanumeric chars)")
    return f"{prefix}-{secrets.token_hex(3)[:5]}"


def is_valid_task_id(task_id: str) -> bool:
    """Validate task ID format."""
    if not task_id or not isinstance(task_id, str):
        return False
    parts = task_id.split('-')
    if len(parts) != 2:
        return False
    prefix, suffix = parts
    return len(prefix) >= 2 and len(suffix) == 5 and suffix.isalnum()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Task:
    """Task with bead-style ID."""
    id: str
    title: str
    description: str = ""
    status: str = "pending"  # pending, active, done, failed, blocked
    depends: List[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    updated: str = ""
    acceptance_criteria: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "depends": self.depends,
            "created": self.created,
            "updated": self.updated or self.created,
            "acceptance_criteria": self.acceptance_criteria,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        return cls(
            id=data.get("id", generate_task_id()),
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=data.get("status", "pending"),
            depends=data.get("depends", []),
            created=data.get("created", datetime.now().isoformat()),
            updated=data.get("updated", ""),
            acceptance_criteria=data.get("acceptance_criteria", []),
            notes=data.get("notes", "")
        )
    
    @classmethod
    def from_user_story(cls, story: dict, prefix: str = "oh") -> 'Task':
        """Convert old prd.json user story to Task."""
        task_id = generate_task_id(prefix)
        status = "done" if story.get("passes", False) else "pending"
        
        return cls(
            id=task_id,
            title=story.get("title", ""),
            description=story.get("description", ""),
            status=status,
            acceptance_criteria=story.get("acceptanceCriteria", []),
            notes=story.get("notes", "")
        )


@dataclass 
class GitCheckpoint:
    """Checkpoint stored as git tag."""
    iteration: int
    task_id: str
    status: str  # in_progress, completed, failed
    timestamp: str
    commit: str = ""  # git commit hash
    
    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "task_id": self.task_id,
            "status": self.status,
            "timestamp": self.timestamp,
            "commit": self.commit
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GitCheckpoint':
        return cls(
            iteration=data.get("iteration", 0),
            task_id=data.get("task_id", ""),
            status=data.get("status", "unknown"),
            timestamp=data.get("timestamp", ""),
            commit=data.get("commit", "")
        )


# =============================================================================
# EXCEPTIONS
# =============================================================================

class GitCommandError(Exception):
    """Exception raised when a git command fails."""
    def __init__(self, message: str, returncode: int = None, stderr: str = None):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class TaskFileCorruptedError(Exception):
    """Raised when the tasks file is corrupted."""
    pass


# =============================================================================
# GIT STATE MANAGER
# =============================================================================

class GitStateManager:
    """
    Git-native state management.
    
    Replaces file-based StateManager with git refs, tags, and notes.
    
    State locations:
    - refs/ralph/iteration: current iteration number
    - refs/ralph/task: current task ID
    - refs/ralph/status: running/paused/stopped
    - ralph/cp/iter-N: checkpoint tags
    - refs/notes/learnings: learnings attached to commits
    - refs/notes/handoff: handoff messages
    - refs/notes/context: task context
    """
    
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self._validate_git_repo()
    
    def _validate_git_repo(self):
        """Ensure workspace is a git repository."""
        git_dir = self.workspace / ".git"
        if not git_dir.exists():
            raise ValueError(f"Not a git repository: {self.workspace}")
    
    def _run_git(self, *args, check: bool = True, capture: bool = True, 
                strict: bool = False) -> subprocess.CompletedProcess:
        """Run git command in workspace.
        
        Args:
            args: Git command arguments
            check: If True, log warning on non-zero exit (does NOT raise)
            capture: Capture stdout/stderr
            strict: If True, raise GitCommandError on non-zero exit
        
        Returns:
            CompletedProcess result
        
        Raises:
            GitCommandError: If strict=True and command fails
            subprocess.TimeoutExpired: If command times out
        
        Note: check=True only logs, it does NOT raise (for backward compat).
              Use strict=True if you need an exception on failure.
        """
        cmd = ["git"] + list(args)
        try:
            result = subprocess.run(
                cmd,
                cwd=self.workspace,
                capture_output=capture,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                if strict:
                    raise GitCommandError(
                        f"Git command failed: {' '.join(cmd)}",
                        returncode=result.returncode,
                        stderr=result.stderr
                    )
                elif check:
                    logger.warning(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
            return result
        except subprocess.TimeoutExpired as e:
            logger.error(f"Git command timed out: {' '.join(cmd)}")
            # Kill the process if still running
            if hasattr(e, 'process') and e.process:
                try:
                    e.process.kill()
                    e.process.wait()
                except Exception:
                    pass
            raise
        except GitCommandError:
            raise  # Re-raise our own exception
        except Exception as e:
            logger.error(f"Git command error: {e}")
            raise
    
    # =========================================================================
    # SECURITY HELPERS
    # =========================================================================
    
    def _validate_ref_name(self, ref_name: str) -> bool:
        """Validate ref name to prevent path traversal."""
        if not ref_name:
            return False
        # Only allow alphanumeric, dash, underscore, forward slash
        if not all(c.isalnum() or c in '-_/' for c in ref_name):
            return False
        # Prevent path traversal
        if '..' in ref_name or ref_name.startswith('/'):
            return False
        # Max length
        if len(ref_name) > 200:
            return False
        return True
    
    def _validate_tag_name(self, tag_name: str) -> bool:
        """Validate git tag name."""
        if not tag_name or len(tag_name) > 250:
            return False
        # Git tag names: alphanumeric, dash, underscore, forward slash
        if not all(c.isalnum() or c in '-_/' for c in tag_name):
            return False
        if '..' in tag_name:
            return False
        return True
    
    def _sanitize_message(self, msg: str, max_length: int = 1000) -> str:
        """Sanitize string for git commit/tag message."""
        if not msg:
            return ""
        # Truncate
        msg = msg[:max_length]
        # Remove null bytes and control chars (except newline)
        msg = ''.join(c for c in msg if c == '\n' or (ord(c) >= 32 and ord(c) != 127))
        return msg
    
    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path is within workspace."""
        if not file_path:
            return False
        try:
            resolved = (self.workspace / file_path).resolve()
            workspace_resolved = self.workspace.resolve()
            return str(resolved).startswith(str(workspace_resolved) + '/')
        except Exception:
            return False
    
    # =========================================================================
    # STATE VIA GIT REFS
    # =========================================================================
    
    def _write_ref(self, ref_name: str, value: str) -> bool:
        """Write value to a git ref file (not a real ref, just storage).
        
        SECURITY: Validates ref_name to prevent path traversal.
        """
        # SECURITY FIX: Validate ref name
        if not self._validate_ref_name(ref_name):
            logger.error(f"Invalid ref name (possible path traversal): {ref_name}")
            return False
        
        ref_path = (self.workspace / ".git" / ref_name).resolve()
        git_dir = (self.workspace / ".git").resolve()
        
        # SECURITY FIX: Ensure path is under .git
        if not str(ref_path).startswith(str(git_dir) + '/'):
            logger.error(f"Ref path escapes .git directory: {ref_path}")
            return False
        
        try:
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_text(value)
            return True
        except Exception as e:
            logger.error(f"Failed to write ref {ref_name}: {e}")
            return False
    
    def _read_ref(self, ref_name: str, default: str = "") -> str:
        """Read value from a git ref file.
        
        SECURITY: Validates ref_name to prevent path traversal.
        """
        # SECURITY FIX: Validate ref name
        if not self._validate_ref_name(ref_name):
            logger.error(f"Invalid ref name (possible path traversal): {ref_name}")
            return default
        
        ref_path = (self.workspace / ".git" / ref_name).resolve()
        git_dir = (self.workspace / ".git").resolve()
        
        # SECURITY FIX: Ensure path is under .git
        if not str(ref_path).startswith(str(git_dir) + '/'):
            logger.error(f"Ref path escapes .git directory: {ref_path}")
            return default
        
        try:
            if ref_path.exists():
                return ref_path.read_text().strip()
        except Exception as e:
            logger.error(f"Failed to read ref {ref_name}: {e}")
        return default
    
    def get_iteration(self) -> int:
        """Get current iteration number."""
        value = self._read_ref("ralph/iteration", "0")
        try:
            return int(value)
        except ValueError:
            return 0
    
    def set_iteration(self, n: int) -> bool:
        """Set current iteration number."""
        return self._write_ref("ralph/iteration", str(n))
    
    def increment_iteration(self) -> int:
        """Atomically increment and return new iteration number.
        
        Uses file locking to prevent race conditions when multiple
        processes try to increment simultaneously.
        """
        import fcntl
        
        ref_path = self.workspace / ".git" / "ralph" / "iteration"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use file locking for atomicity
        try:
            with open(ref_path, 'a+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    content = f.read().strip()
                    n = int(content) if content else 0
                    n += 1
                    f.seek(0)
                    f.truncate()
                    f.write(str(n))
                    f.flush()
                    return n
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to increment iteration: {e}")
            # Fallback to non-atomic increment
            n = self.get_iteration() + 1
            self.set_iteration(n)
            return n
    
    def get_current_task(self) -> str:
        """Get current task ID."""
        return self._read_ref("ralph/task", "")
    
    def set_current_task(self, task_id: str) -> bool:
        """Set current task ID."""
        return self._write_ref("ralph/task", task_id)
    
    def get_status(self) -> str:
        """Get Ralph status: running, paused, stopped."""
        return self._read_ref("ralph/status", "stopped")
    
    def set_status(self, status: str) -> bool:
        """Set Ralph status."""
        if status not in ("running", "paused", "stopped"):
            raise ValueError(f"Invalid status: {status}")
        return self._write_ref("ralph/status", status)
    
    # =========================================================================
    # CHECKPOINTS VIA GIT TAGS
    # =========================================================================
    
    def save_checkpoint(self, iteration: int, task_id: str, status: str) -> bool:
        """Save checkpoint as git tag."""
        tag_name = f"ralph/cp/iter-{iteration}"
        
        checkpoint = GitCheckpoint(
            iteration=iteration,
            task_id=task_id,
            status=status,
            timestamp=datetime.now().isoformat(),
            commit=self._get_head_commit()
        )
        
        # Create annotated tag with JSON message
        message = json.dumps(checkpoint.to_dict())
        
        # Delete existing tag if exists
        self._run_git("tag", "-d", tag_name, check=False)
        
        # Create new tag
        result = self._run_git("tag", "-a", tag_name, "-m", message)
        return result.returncode == 0
    
    def load_checkpoint(self, iteration: Optional[int] = None) -> Optional[GitCheckpoint]:
        """Load checkpoint from git tag."""
        if iteration is None:
            # Find latest checkpoint
            result = self._run_git("tag", "-l", "ralph/cp/iter-*", "--sort=-version:refname")
            if result.returncode != 0 or not result.stdout.strip():
                return None
            tag_name = result.stdout.strip().split('\n')[0]
        else:
            tag_name = f"ralph/cp/iter-{iteration}"
        
        # Get tag message
        result = self._run_git("tag", "-l", tag_name, "-n1000", "--format=%(contents)")
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        try:
            data = json.loads(result.stdout.strip())
            return GitCheckpoint.from_dict(data)
        except json.JSONDecodeError:
            logger.error(f"Invalid checkpoint JSON in tag {tag_name}")
            return None
    
    def list_checkpoints(self, limit: int = 20) -> List[GitCheckpoint]:
        """List recent checkpoints."""
        result = self._run_git("tag", "-l", "ralph/cp/iter-*", "--sort=-version:refname")
        if result.returncode != 0 or not result.stdout.strip():
            return []
        
        checkpoints = []
        for tag_name in result.stdout.strip().split('\n')[:limit]:
            cp = self.load_checkpoint_by_tag(tag_name)
            if cp:
                checkpoints.append(cp)
        
        return checkpoints
    
    def load_checkpoint_by_tag(self, tag_name: str) -> Optional[GitCheckpoint]:
        """Load checkpoint by tag name."""
        result = self._run_git("tag", "-l", tag_name, "-n1000", "--format=%(contents)")
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        try:
            data = json.loads(result.stdout.strip())
            return GitCheckpoint.from_dict(data)
        except json.JSONDecodeError:
            return None
    
    # =========================================================================
    # ITERATIONS VIA COMMIT MESSAGES
    # =========================================================================
    
    def commit_iteration(self, iteration: int, task_id: str, summary: str, 
                        files: List[str] = None) -> bool:
        """
        Commit iteration progress with structured message.
        
        Message format: [Ralph:Iter:N] task-id: summary
        
        SECURITY: Validates file paths and sanitizes message content.
        """
        # Stage files
        if files:
            for f in files:
                # SECURITY FIX: Validate file path is within workspace
                if not self._validate_file_path(f):
                    logger.warning(f"Skipping file outside workspace: {f}")
                    continue
                # Use relative path for git add
                try:
                    file_path = (self.workspace / f).resolve()
                    rel_path = file_path.relative_to(self.workspace.resolve())
                    self._run_git("add", str(rel_path), check=False)
                except ValueError:
                    logger.warning(f"Cannot resolve relative path: {f}")
                    continue
        else:
            # Stage all changes in .ralph/
            self._run_git("add", ".ralph/", check=False)
        
        # Check if there are changes to commit
        result = self._run_git("status", "--porcelain")
        if not result.stdout.strip():
            logger.debug("No changes to commit for iteration")
            return True  # No changes is OK
        
        # SECURITY FIX: Sanitize message components
        safe_task_id = self._sanitize_message(str(task_id), 100)
        safe_summary = self._sanitize_message(str(summary), 500)
        
        # Commit with structured message
        message = f"[Ralph:Iter:{iteration}] {safe_task_id}: {safe_summary}"
        result = self._run_git("commit", "-m", message, "--allow-empty")
        return result.returncode == 0
    
    def get_iteration_history(self, limit: int = 50) -> List[dict]:
        """Get iteration history from git log."""
        result = self._run_git(
            "log", 
            f"-{limit}",
            "--grep=[Ralph:Iter:",
            "--format=%H|%s|%ai"
        )
        
        if result.returncode != 0 or not result.stdout.strip():
            return []
        
        history = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split('|', 2)
            if len(parts) == 3:
                commit_hash, message, date = parts
                # Parse message: [Ralph:Iter:N] task-id: summary
                try:
                    iter_part = message.split(']')[0].replace('[Ralph:Iter:', '')
                    iteration = int(iter_part)
                    rest = message.split(']', 1)[1].strip()
                    task_id = rest.split(':')[0].strip()
                    summary = ':'.join(rest.split(':')[1:]).strip()
                    
                    history.append({
                        "commit": commit_hash,
                        "iteration": iteration,
                        "task_id": task_id,
                        "summary": summary,
                        "date": date
                    })
                except (ValueError, IndexError):
                    continue
        
        return history
    
    # =========================================================================
    # LEARNINGS VIA GIT NOTES
    # =========================================================================
    
    def add_learning(self, learning: str, commit: str = "HEAD") -> bool:
        """Add learning as git note."""
        timestamp = datetime.now().isoformat()
        note = f"[{timestamp}] {learning}"
        
        # Append to existing notes
        result = self._run_git("notes", "--ref=learnings", "append", "-m", note, commit)
        return result.returncode == 0
    
    def get_learnings(self, commit: str = "HEAD", limit: int = 100) -> List[str]:
        """Get learnings from git notes."""
        # Get commits with learnings notes
        result = self._run_git(
            "log",
            f"-{limit}",
            "--format=%H",
            "--notes=learnings"
        )
        
        if result.returncode != 0:
            return []
        
        learnings = []
        for commit_hash in result.stdout.strip().split('\n'):
            if not commit_hash:
                continue
            note_result = self._run_git("notes", "--ref=learnings", "show", commit_hash, check=False)
            if note_result.returncode == 0 and note_result.stdout.strip():
                for line in note_result.stdout.strip().split('\n'):
                    if line.strip():
                        learnings.append(line.strip())
        
        return learnings[:limit]
    
    def get_all_learnings_text(self, limit: int = 100) -> str:
        """Get all learnings as formatted text."""
        learnings = self.get_learnings(limit=limit)
        if not learnings:
            return ""
        return "\n".join(f"- {l}" for l in learnings)
    
    # =========================================================================
    # HANDOFF VIA GIT NOTES
    # =========================================================================
    
    def write_handoff(self, task_id: str, message: str, context: dict = None) -> bool:
        """Write handoff message for next iteration."""
        handoff = {
            "task_id": task_id,
            "message": message,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        note = json.dumps(handoff)
        result = self._run_git("notes", "--ref=handoff", "add", "-f", "-m", note, "HEAD")
        return result.returncode == 0
    
    def read_handoff(self) -> Optional[dict]:
        """Read handoff from previous iteration."""
        result = self._run_git("notes", "--ref=handoff", "show", "HEAD", check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return None
    
    def clear_handoff(self) -> bool:
        """Clear handoff after reading."""
        result = self._run_git("notes", "--ref=handoff", "remove", "HEAD", check=False)
        return True  # OK even if no note exists
    
    # =========================================================================
    # TASK CONTEXT VIA GIT NOTES
    # =========================================================================
    
    def save_task_context(self, task_id: str, context: dict) -> bool:
        """Save task context as git note."""
        context["task_id"] = task_id
        context["updated"] = datetime.now().isoformat()
        
        note = json.dumps(context)
        result = self._run_git("notes", "--ref=context", "add", "-f", "-m", note, "HEAD")
        return result.returncode == 0
    
    def get_task_context(self, task_id: str = None) -> Optional[dict]:
        """Get task context from git notes."""
        result = self._run_git("notes", "--ref=context", "show", "HEAD", check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        try:
            context = json.loads(result.stdout.strip())
            if task_id and context.get("task_id") != task_id:
                return None
            return context
        except json.JSONDecodeError:
            return None
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _get_head_commit(self) -> str:
        """Get HEAD commit hash."""
        result = self._run_git("rev-parse", "HEAD", check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    
    def get_state_summary(self) -> dict:
        """Get complete state summary."""
        # FIX: Load checkpoint once, not twice
        checkpoint = self.load_checkpoint()
        return {
            "iteration": self.get_iteration(),
            "task": self.get_current_task(),
            "status": self.get_status(),
            "head": self._get_head_commit(),
            "checkpoint": checkpoint.to_dict() if checkpoint else None,
            "learnings_count": len(self.get_learnings(limit=1000))
        }
    
    def reset_state(self) -> bool:
        """Reset all Ralph state (for fresh start)."""
        try:
            self.set_iteration(0)
            self.set_current_task("")
            self.set_status("stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to reset state: {e}")
            return False


# =============================================================================
# TASK MANAGER (Git-Native with bead-style IDs)
# =============================================================================

class TaskManager:
    """
    Manages tasks with bead-style IDs.
    
    v5.1: Tasks stored in git blob via refs/ralph/tasks (NO FILE).
    Replaces old prd.json and tasks.json with pure git storage.
    
    Storage:
        refs/ralph/tasks -> blob containing JSON
        
    History:
        git reflog refs/ralph/tasks  # See all changes
        
    Benefits:
        - No file race conditions (git handles locking)
        - Full history via reflog
        - Atomic updates
        - Crash-safe
    """
    
    VERSION = 3  # Task format version (v3 = git-native)
    TASKS_REF = "refs/ralph/tasks"
    
    def __init__(self, workspace: Path, project_prefix: str = "oh", 
                 ralph_dir: Optional[Path] = None):
        """
        Args:
            workspace: Git repository root (contains .git/)
            project_prefix: Prefix for task IDs (e.g., "oh" -> "oh-k7m2x")
            ralph_dir: Optional .ralph/ dir for migration (can be None after migration)
        """
        self.workspace = Path(workspace)
        self.ralph_dir = Path(ralph_dir) if ralph_dir else self.workspace / ".ralph"
        self.project_prefix = project_prefix
        self._tasks: Dict[str, Task] = {}
        
        # Validate git repo exists
        if not (self.workspace / ".git").exists():
            raise ValueError(f"Not a git repository: {self.workspace}")
        
        self._load()
    
    def _run_git(self, *args, input_data: str = None) -> Tuple[bool, str]:
        """Run git command, return (success, output)."""
        cmd = ["git"] + list(args)
        try:
            result = subprocess.run(
                cmd,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                input=input_data,
                timeout=30
            )
            return result.returncode == 0, result.stdout.strip()
        except Exception as e:
            logger.error(f"Git command failed: {cmd}: {e}")
            return False, str(e)
    
    def _load(self):
        """Load tasks from git blob.
        
        Tries in order:
        1. Git blob at refs/ralph/tasks (v3 format)
        2. File .ralph/tasks.json (v2 format) -> migrate
        3. File .ralph/prd.json (v1 format) -> migrate
        4. Empty (new project)
        """
        # Try git blob first
        success, blob_content = self._run_git("show", self.TASKS_REF)
        if success and blob_content.strip():
            try:
                data = json.loads(blob_content)
                
                # HIGH-1 FIX: Validate required structure
                if not isinstance(data, dict):
                    raise TaskFileCorruptedError("Tasks blob is not a dict")
                if "tasks" not in data:
                    raise TaskFileCorruptedError("Tasks blob missing 'tasks' key")
                
                for task_id, task_data in data.get("tasks", {}).items():
                    # Validate required fields
                    if not isinstance(task_data, dict):
                        logger.warning(f"Task {task_id} is not a dict, skipping")
                        continue
                    if "title" not in task_data:
                        logger.warning(f"Task {task_id} missing 'title', skipping")
                        continue
                    
                    task_data["id"] = task_id
                    # Set defaults for optional fields
                    task_data.setdefault("status", "pending")
                    task_data.setdefault("description", "")
                    task_data.setdefault("depends", [])
                    
                    self._tasks[task_id] = Task.from_dict(task_data)
                
                logger.debug(f"Loaded {len(self._tasks)} tasks from git blob")
                return
            except json.JSONDecodeError as e:
                logger.error(f"Corrupted tasks blob: {e}")
                raise TaskFileCorruptedError(
                    f"Tasks blob corrupted at {self.TASKS_REF}. "
                    f"Check git reflog refs/ralph/tasks for history."
                ) from e
        
        # Try file-based migration (v2 tasks.json)
        tasks_file = self.ralph_dir / "tasks.json"
        if tasks_file.exists():
            self._migrate_from_file(tasks_file)
            return
        
        # Try file-based migration (v1 prd.json)
        prd_file = self.ralph_dir / "prd.json"
        if prd_file.exists():
            self._migrate_from_prd(prd_file)
            return
        
        # New project - empty tasks
        self._tasks = {}
        logger.debug("No existing tasks found, starting fresh")
    
    def _migrate_from_file(self, tasks_file: Path):
        """Migrate from tasks.json (v2) to git blob (v3)."""
        logger.info(f"Migrating tasks from {tasks_file} to git blob")
        
        try:
            data = json.loads(tasks_file.read_text())
            for task_id, task_data in data.get("tasks", {}).items():
                task_data["id"] = task_id
                # Set defaults for missing fields
                task_data.setdefault("status", "pending")
                task_data.setdefault("description", "")
                task_data.setdefault("depends", [])
                self._tasks[task_id] = Task.from_dict(task_data)
            
            # Save to git blob
            if not self._save():
                raise RuntimeError("Failed to save tasks to git blob")
            
            # HIGH-4 FIX: Verify save actually worked before deleting original
            success, verify_content = self._run_git("show", self.TASKS_REF)
            if not success or not verify_content.strip():
                raise RuntimeError("Migration save verification failed - ref not readable")
            
            # Only then backup and remove old file
            backup = tasks_file.with_suffix('.json.migrated')
            tasks_file.rename(backup)
            logger.info(f"Migrated {len(self._tasks)} tasks, old file moved to {backup.name}")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    def _migrate_from_prd(self, prd_file: Path):
        """Migrate from prd.json (v1) to git blob (v3)."""
        logger.info(f"Migrating from prd.json to git blob")
        
        try:
            data = json.loads(prd_file.read_text())
            stories = data.get("userStories", [])
            
            for story in stories:
                task = Task.from_user_story(story, self.project_prefix)
                self._tasks[task.id] = task
            
            # Save to git blob
            if not self._save():
                raise RuntimeError("Failed to save tasks to git blob")
            
            # HIGH-4 FIX: Verify save actually worked before deleting original
            success, verify_content = self._run_git("show", self.TASKS_REF)
            if not success or not verify_content.strip():
                raise RuntimeError("Migration save verification failed - ref not readable")
            
            # Only then backup and remove old file
            backup = prd_file.with_suffix('.json.migrated')
            prd_file.rename(backup)
            logger.info(f"Migrated {len(stories)} stories, old file moved to {backup.name}")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    def _save(self, retry_count: int = 0) -> bool:
        """Save tasks to git blob atomically with compare-and-swap.
        
        Creates a blob with JSON content and updates refs/ralph/tasks.
        Uses CAS to detect concurrent modifications.
        
        HIGH-2 FIX: Compare-and-swap prevents lost updates from races.
        """
        MAX_RETRIES = 3
        
        try:
            # Get current ref value for CAS
            _, old_ref_output = self._run_git("show-ref", self.TASKS_REF)
            old_hash = old_ref_output.split()[0] if old_ref_output.strip() else ""
            
            data = {
                "version": self.VERSION,
                "project": self.project_prefix,
                "tasks": {tid: t.to_dict() for tid, t in self._tasks.items()},
                "active_task": self.get_active_task_id(),
                "updated": datetime.now().isoformat()
            }
            
            json_content = json.dumps(data, indent=2)
            
            # Create blob: echo content | git hash-object -w --stdin
            success, blob_hash = self._run_git(
                "hash-object", "-w", "--stdin",
                input_data=json_content
            )
            if not success:
                logger.error(f"Failed to create blob: {blob_hash}")
                return False
            
            # Update ref with CAS (old_hash as expected value)
            if old_hash:
                # Existing ref - use CAS
                success, msg = self._run_git(
                    "update-ref", self.TASKS_REF, blob_hash, old_hash
                )
            else:
                # New ref - just create it
                success, msg = self._run_git(
                    "update-ref", self.TASKS_REF, blob_hash
                )
            
            if not success:
                if retry_count < MAX_RETRIES:
                    logger.warning(f"Concurrent modification detected, retrying ({retry_count + 1}/{MAX_RETRIES})...")
                    # Reload and merge
                    self._load()
                    return self._save(retry_count + 1)
                else:
                    logger.error(f"Failed to update ref after {MAX_RETRIES} retries: {msg}")
                    return False
            
            logger.debug(f"Saved {len(self._tasks)} tasks to git blob {blob_hash[:8]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")
            return False
    
    def create_task(self, title: str, description: str = "", 
                   depends: List[str] = None,
                   acceptance_criteria: List[str] = None) -> Task:
        """Create new task with generated ID."""
        task = Task(
            id=generate_task_id(self.project_prefix),
            title=title,
            description=description,
            depends=depends or [],
            acceptance_criteria=acceptance_criteria or []
        )
        self._tasks[task.id] = task
        self._save()
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)
    
    def update_task(self, task_id: str, **kwargs) -> bool:
        """Update task fields."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        task.updated = datetime.now().isoformat()
        self._save()
        return True
    
    def set_task_status(self, task_id: str, status: str) -> bool:
        """Set task status."""
        return self.update_task(task_id, status=status)
    
    def get_active_task_id(self) -> Optional[str]:
        """Get currently active task ID."""
        for task in self._tasks.values():
            if task.status == "active":
                return task.id
        return None
    
    def get_active_task(self) -> Optional[Task]:
        """Get currently active task."""
        task_id = self.get_active_task_id()
        return self._tasks.get(task_id) if task_id else None
    
    def set_active_task(self, task_id: str) -> bool:
        """Set task as active, deactivate others."""
        if task_id not in self._tasks:
            return False
        
        for tid, task in self._tasks.items():
            if task.status == "active":
                task.status = "pending"
        
        self._tasks[task_id].status = "active"
        self._save()
        return True
    
    def get_next_task(self) -> Optional[Task]:
        """Get next pending task (respecting dependencies).
        
        Detects circular dependencies and treats them as unmet.
        """
        for task in self._tasks.values():
            if task.status != "pending":
                continue
            
            # Check dependencies with cycle detection
            if self._deps_met(task.id, set()):
                return task
        
        return None
    
    def _deps_met(self, task_id: str, visited: set) -> bool:
        """Check if dependencies are met, detecting cycles.
        
        Args:
            task_id: Task to check
            visited: Set of already visited task IDs (for cycle detection)
        
        Returns:
            True if all dependencies are satisfied
        """
        # Cycle detection
        if task_id in visited:
            logger.warning(f"Circular dependency detected involving: {task_id}")
            return False
        
        task = self._tasks.get(task_id)
        if not task:
            return True  # Missing task = assume satisfied
        
        visited.add(task_id)
        
        for dep_id in task.depends:
            dep = self._tasks.get(dep_id)
            if not dep:
                continue  # Missing dependency = ignore
            if dep.status != "done":
                return False
            # Check transitive dependencies (recursive)
            if not self._deps_met(dep_id, visited.copy()):
                return False
        
        return True
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self._tasks.values())
    
    def get_tasks_by_status(self, status: str) -> List[Task]:
        """Get tasks by status."""
        return [t for t in self._tasks.values() if t.status == status]
    
    def get_progress(self) -> Tuple[int, int]:
        """Get (done, total) count."""
        done = len(self.get_tasks_by_status("done"))
        total = len(self._tasks)
        return (done, total)
    
    def delete_task(self, task_id: str) -> bool:
        """Delete task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._save()
            return True
        return False
    
    def to_prompt_format(self) -> str:
        """Format tasks for LLM prompt."""
        lines = ["## Tasks\n"]
        
        for task in self._tasks.values():
            status_emoji = {
                "done": "✅",
                "active": "🔄", 
                "pending": "⏳",
                "failed": "❌",
                "blocked": "🚫"
            }.get(task.status, "❓")
            
            lines.append(f"### {status_emoji} {task.id}: {task.title}")
            lines.append(f"Status: {task.status}")
            
            if task.description:
                lines.append(f"\n{task.description}")
            
            if task.depends:
                lines.append(f"\nDepends on: {', '.join(task.depends)}")
            
            if task.acceptance_criteria:
                lines.append("\nAcceptance criteria:")
                for ac in task.acceptance_criteria:
                    lines.append(f"  - {ac}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    # =========================================================================
    # MIGRATION FROM OLD FORMAT
    # =========================================================================
    
    @classmethod
    def migrate_prd_file(cls, prd_file: Path, ralph_dir: Path, 
                        prefix: str = "oh") -> 'TaskManager':
        """
        Migrate old prd.json to new tasks.json format.
        
        Preserves old file as prd.json.bak
        """
        if not prd_file.exists():
            return cls(ralph_dir, prefix)
        
        # Backup old file
        backup_file = prd_file.with_suffix('.json.bak')
        if not backup_file.exists():
            import shutil
            shutil.copy(prd_file, backup_file)
        
        # Load and convert
        try:
            data = json.loads(prd_file.read_text())
            
            manager = cls(ralph_dir, prefix)
            
            # Convert user stories
            for story in data.get("userStories", []):
                task = Task.from_user_story(story, prefix)
                manager._tasks[task.id] = task
            
            manager._save()
            
            # Remove old prd.json (keep backup)
            prd_file.unlink()
            
            logger.info(f"Migrated {len(manager._tasks)} tasks from prd.json")
            return manager
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return cls(ralph_dir, prefix)


# =============================================================================
# FORMULA SYSTEM (TOML workflows)
# =============================================================================

@dataclass
class FormulaStep:
    """Single step in a formula."""
    id: str
    title: str
    description: str
    needs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "needs": self.needs
        }


@dataclass
class Formula:
    """TOML-defined workflow formula."""
    name: str
    description: str
    steps: List[FormulaStep]
    version: int = 1
    vars: Dict[str, dict] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "vars": self.vars,
            "steps": [s.to_dict() for s in self.steps]
        }
    
    @classmethod
    def from_toml(cls, content: str) -> 'Formula':
        """Parse formula from TOML content."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        data = tomllib.loads(content)
        
        steps = []
        for step_data in data.get("steps", []):
            steps.append(FormulaStep(
                id=step_data.get("id", ""),
                title=step_data.get("title", ""),
                description=step_data.get("description", ""),
                needs=step_data.get("needs", [])
            ))
        
        return cls(
            name=data.get("formula", data.get("name", "unnamed")),
            description=data.get("description", ""),
            version=data.get("version", 1),
            vars=data.get("vars", {}),
            steps=steps
        )


class FormulaManager:
    """
    Manages TOML workflow formulas.
    
    Formulas define reusable task sequences.
    
    v5.1: Formulas now in workspace/formulas/ (visible, editable)
    instead of .ralph/formulas/ (hidden).
    """
    
    def __init__(self, workspace: Path, ralph_dir: Optional[Path] = None):
        """
        Args:
            workspace: Git repository root
            ralph_dir: Optional .ralph/ dir for migration
        """
        self.workspace = Path(workspace)
        self.ralph_dir = Path(ralph_dir) if ralph_dir else self.workspace / ".ralph"
        
        # New location: workspace/formulas/ (visible)
        self.formulas_dir = self.workspace / "formulas"
        self.formulas_dir.mkdir(parents=True, exist_ok=True)
        
        # Migrate from old location if exists
        old_formulas_dir = self.ralph_dir / "formulas"
        if old_formulas_dir.exists() and old_formulas_dir.is_dir():
            self._migrate_formulas(old_formulas_dir)
    
    def _migrate_formulas(self, old_dir: Path):
        """Migrate formulas from .ralph/formulas/ to workspace/formulas/."""
        import shutil
        migrated = 0
        for f in old_dir.glob("*.toml"):
            dest = self.formulas_dir / f.name
            if not dest.exists():
                shutil.copy(f, dest)
                migrated += 1
        
        if migrated > 0:
            logger.info(f"Migrated {migrated} formulas from {old_dir} to {self.formulas_dir}")
            # Rename old directory to mark as migrated
            old_dir.rename(old_dir.with_suffix('.migrated'))
    
    def list_formulas(self) -> List[str]:
        """List available formula names."""
        formulas = []
        for f in self.formulas_dir.glob("*.toml"):
            formulas.append(f.stem)
        return formulas
    
    def get_formula(self, name: str) -> Optional[Formula]:
        """Load formula by name."""
        formula_file = self.formulas_dir / f"{name}.toml"
        if not formula_file.exists():
            return None
        
        try:
            return Formula.from_toml(formula_file.read_text())
        except Exception as e:
            logger.error(f"Failed to load formula {name}: {e}")
            return None
    
    def cook_formula(self, name: str, task_manager: TaskManager,
                    vars: Dict[str, str] = None) -> List[Task]:
        """
        Execute formula: create tasks from formula steps.
        
        Args:
            name: Formula name
            task_manager: TaskManager to create tasks in
            vars: Variable substitutions
        
        Returns:
            List of created tasks
        """
        formula = self.get_formula(name)
        if not formula:
            raise ValueError(f"Formula not found: {name}")
        
        vars = vars or {}
        
        # Validate required vars
        for var_name, var_def in formula.vars.items():
            if var_def.get("required", False) and var_name not in vars:
                raise ValueError(f"Required variable not provided: {var_name}")
        
        # Map step IDs to task IDs
        step_to_task: Dict[str, str] = {}
        created_tasks: List[Task] = []
        
        for step in formula.steps:
            # Substitute variables in description
            description = step.description
            for var_name, var_value in vars.items():
                description = description.replace(f"{{{{{var_name}}}}}", var_value)
            
            # Map step dependencies to task IDs
            depends = [step_to_task[dep] for dep in step.needs if dep in step_to_task]
            
            # Create task
            task = task_manager.create_task(
                title=step.title,
                description=description,
                depends=depends
            )
            
            step_to_task[step.id] = task.id
            created_tasks.append(task)
        
        return created_tasks
    
    def save_formula(self, formula: Formula) -> bool:
        """Save formula to TOML file."""
        try:
            import tomli_w
        except ImportError:
            # Fallback to manual TOML generation
            return self._save_formula_manual(formula)
        
        formula_file = self.formulas_dir / f"{formula.name}.toml"
        
        try:
            content = tomli_w.dumps(formula.to_dict())
            formula_file.write_text(content)
            return True
        except Exception as e:
            logger.error(f"Failed to save formula: {e}")
            return False
    
    def _save_formula_manual(self, formula: Formula) -> bool:
        """Save formula without tomli_w.
        
        Uses TOML multiline strings (triple quotes) for descriptions
        that may contain newlines.
        """
        formula_file = self.formulas_dir / f"{formula.name}.toml"
        
        def escape_string(s: str) -> str:
            """Escape string for TOML, using multiline if needed."""
            if '\n' in s:
                # Use multiline literal string (triple single quotes)
                # Replace any ''' in content to avoid breaking
                safe = s.replace("'''", "' ' '")
                return f"'''\n{safe}'''"
            else:
                # Simple string with escaped quotes
                return f'"{s.replace(chr(34), chr(92)+chr(34))}"'
        
        lines = [
            f'description = {escape_string(formula.description)}',
            f'formula = "{formula.name}"',
            f'version = {formula.version}',
            ''
        ]
        
        for var_name, var_def in formula.vars.items():
            lines.append(f'[vars.{var_name}]')
            for k, v in var_def.items():
                if isinstance(v, bool):
                    lines.append(f'{k} = {str(v).lower()}')
                elif isinstance(v, str):
                    lines.append(f'{k} = {escape_string(v)}')
                else:
                    lines.append(f'{k} = {v}')
            lines.append('')
        
        for step in formula.steps:
            lines.append('[[steps]]')
            lines.append(f'id = "{step.id}"')
            lines.append(f'title = {escape_string(step.title)}')
            lines.append(f'description = {escape_string(step.description)}')
            if step.needs:
                needs_str = ', '.join(f'"{n}"' for n in step.needs)
                lines.append(f'needs = [{needs_str}]')
            lines.append('')
        
        try:
            formula_file.write_text('\n'.join(lines))
            return True
        except Exception as e:
            logger.error(f"Failed to save formula: {e}")
            return False
    
    def create_builtin_formulas(self):
        """Create built-in formulas."""
        
        # Bugfix formula
        bugfix = Formula(
            name="bugfix",
            description="Standard bug fix workflow",
            vars={
                "bug_description": {
                    "description": "Description of the bug",
                    "required": True
                }
            },
            steps=[
                FormulaStep(
                    id="reproduce",
                    title="Reproduce the bug",
                    description="{{bug_description}}\n\nWrite a failing test that reproduces this bug."
                ),
                FormulaStep(
                    id="fix",
                    title="Implement the fix",
                    description="Fix the bug so the test passes.",
                    needs=["reproduce"]
                ),
                FormulaStep(
                    id="verify",
                    title="Verify fix",
                    description="Run all tests, ensure no regressions.",
                    needs=["fix"]
                )
            ]
        )
        self.save_formula(bugfix)
        
        # Feature formula
        feature = Formula(
            name="feature",
            description="Standard feature development workflow",
            vars={
                "feature_name": {"description": "Name of the feature", "required": True},
                "feature_description": {"description": "Detailed description", "required": True}
            },
            steps=[
                FormulaStep(
                    id="design",
                    title="Design {{feature_name}}",
                    description="Design the implementation approach for:\n\n{{feature_description}}"
                ),
                FormulaStep(
                    id="implement",
                    title="Implement {{feature_name}}",
                    description="Implement the designed solution.",
                    needs=["design"]
                ),
                FormulaStep(
                    id="test",
                    title="Test {{feature_name}}",
                    description="Write tests for the new feature.",
                    needs=["implement"]
                ),
                FormulaStep(
                    id="document",
                    title="Document {{feature_name}}",
                    description="Update documentation for the new feature.",
                    needs=["test"]
                )
            ]
        )
        self.save_formula(feature)
        
        # Refactor formula
        refactor = Formula(
            name="refactor",
            description="Code refactoring workflow",
            vars={
                "target": {"description": "Code to refactor", "required": True},
                "goal": {"description": "Refactoring goal", "required": True}
            },
            steps=[
                FormulaStep(
                    id="analyze",
                    title="Analyze current code",
                    description="Analyze {{target}} for refactoring opportunities.\nGoal: {{goal}}"
                ),
                FormulaStep(
                    id="test-before",
                    title="Add characterization tests",
                    description="Add tests to capture current behavior before refactoring.",
                    needs=["analyze"]
                ),
                FormulaStep(
                    id="refactor",
                    title="Perform refactoring",
                    description="Refactor the code to achieve: {{goal}}",
                    needs=["test-before"]
                ),
                FormulaStep(
                    id="verify",
                    title="Verify refactoring",
                    description="Run all tests, verify behavior unchanged.",
                    needs=["refactor"]
                )
            ]
        )
        self.save_formula(refactor)
        
        logger.info("Created built-in formulas: bugfix, feature, refactor")
