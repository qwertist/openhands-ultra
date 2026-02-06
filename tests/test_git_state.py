#!/usr/bin/env python3
"""
Tests for git_state.py - Git-native state management

Run: pytest tests/test_git_state.py -v
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from git_state import (
    generate_task_id, is_valid_task_id,
    Task, GitCheckpoint,
    GitStateManager, TaskManager, FormulaManager,
    Formula, FormulaStep
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir) / "workspace"
    workspace.mkdir()
    
    # Initialize git repo
    subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=workspace, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, capture_output=True)
    
    # Create initial commit
    (workspace / "README.md").write_text("# Test")
    subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=workspace, capture_output=True)
    
    # Create .ralph directory
    ralph_dir = workspace / ".ralph"
    ralph_dir.mkdir()
    
    yield workspace, ralph_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def git_state(temp_git_repo):
    """Create GitStateManager instance."""
    workspace, ralph_dir = temp_git_repo
    return GitStateManager(workspace)


@pytest.fixture
def task_manager(temp_git_repo):
    """Create TaskManager instance."""
    workspace, ralph_dir = temp_git_repo
    return TaskManager(workspace, "ts", ralph_dir)


@pytest.fixture
def formula_manager(temp_git_repo):
    """Create FormulaManager instance."""
    workspace, ralph_dir = temp_git_repo
    return FormulaManager(workspace, ralph_dir)


# =============================================================================
# TASK ID TESTS
# =============================================================================

class TestTaskId:
    def test_generate_task_id_format(self):
        """Task ID should be prefix-xxxxx format."""
        task_id = generate_task_id("oh")
        assert task_id.startswith("oh-")
        assert len(task_id) == 8  # "oh-" + 5 chars
    
    def test_generate_task_id_unique(self):
        """Generated IDs should be unique."""
        ids = [generate_task_id() for _ in range(100)]
        assert len(ids) == len(set(ids))
    
    def test_generate_task_id_custom_prefix(self):
        """Custom prefix should be used."""
        task_id = generate_task_id("proj")
        assert task_id.startswith("proj-")
    
    def test_is_valid_task_id_valid(self):
        """Valid IDs should pass validation."""
        assert is_valid_task_id("oh-a1b2c")
        assert is_valid_task_id("proj-12345")
        assert is_valid_task_id("xx-abcde")
    
    def test_is_valid_task_id_invalid(self):
        """Invalid IDs should fail validation."""
        assert not is_valid_task_id("")
        assert not is_valid_task_id(None)
        assert not is_valid_task_id("oh-abc")  # Too short
        assert not is_valid_task_id("oh-abcdef")  # Too long
        assert not is_valid_task_id("o-abcde")  # Prefix too short
        assert not is_valid_task_id("oh_abcde")  # Wrong separator
        assert not is_valid_task_id("abcde")  # No prefix


# =============================================================================
# TASK TESTS
# =============================================================================

class TestTask:
    def test_task_creation(self):
        """Task should be created with default values."""
        task = Task(id="oh-a1b2c", title="Test task")
        assert task.id == "oh-a1b2c"
        assert task.title == "Test task"
        assert task.status == "pending"
        assert task.depends == []
    
    def test_task_to_dict(self):
        """Task should serialize to dict."""
        task = Task(id="oh-a1b2c", title="Test", description="Desc")
        d = task.to_dict()
        assert d["id"] == "oh-a1b2c"
        assert d["title"] == "Test"
        assert d["description"] == "Desc"
    
    def test_task_from_dict(self):
        """Task should deserialize from dict."""
        data = {"id": "oh-a1b2c", "title": "Test", "status": "active"}
        task = Task.from_dict(data)
        assert task.id == "oh-a1b2c"
        assert task.title == "Test"
        assert task.status == "active"
    
    def test_task_from_user_story(self):
        """Task should convert from old prd.json format."""
        story = {
            "id": 1,
            "title": "Old story",
            "description": "Old desc",
            "passes": True,
            "acceptanceCriteria": ["AC1", "AC2"]
        }
        task = Task.from_user_story(story, "oh")
        assert task.title == "Old story"
        assert task.status == "done"
        assert task.acceptance_criteria == ["AC1", "AC2"]


# =============================================================================
# GIT STATE MANAGER TESTS
# =============================================================================

class TestGitStateManager:
    def test_init_valid_repo(self, git_state):
        """Should initialize with valid git repo."""
        assert git_state is not None
    
    def test_init_invalid_repo(self):
        """Should raise error for non-git directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Not a git repository"):
                GitStateManager(Path(temp_dir))
    
    def test_iteration_get_set(self, git_state):
        """Should get/set iteration number."""
        assert git_state.get_iteration() == 0
        git_state.set_iteration(42)
        assert git_state.get_iteration() == 42
    
    def test_iteration_increment(self, git_state):
        """Should increment iteration."""
        git_state.set_iteration(10)
        n = git_state.increment_iteration()
        assert n == 11
        assert git_state.get_iteration() == 11
    
    def test_current_task_get_set(self, git_state):
        """Should get/set current task ID."""
        assert git_state.get_current_task() == ""
        git_state.set_current_task("oh-a1b2c")
        assert git_state.get_current_task() == "oh-a1b2c"
    
    def test_status_get_set(self, git_state):
        """Should get/set status."""
        assert git_state.get_status() == "stopped"
        git_state.set_status("running")
        assert git_state.get_status() == "running"
    
    def test_status_invalid(self, git_state):
        """Should reject invalid status."""
        with pytest.raises(ValueError, match="Invalid status"):
            git_state.set_status("invalid")
    
    def test_checkpoint_save_load(self, git_state):
        """Should save and load checkpoint."""
        git_state.save_checkpoint(42, "oh-a1b2c", "in_progress")
        cp = git_state.load_checkpoint()
        
        assert cp is not None
        assert cp.iteration == 42
        assert cp.task_id == "oh-a1b2c"
        assert cp.status == "in_progress"
    
    def test_checkpoint_latest(self, git_state):
        """Should load latest checkpoint."""
        git_state.save_checkpoint(1, "oh-t1", "completed")
        git_state.save_checkpoint(2, "oh-t2", "completed")
        git_state.save_checkpoint(3, "oh-t3", "in_progress")
        
        cp = git_state.load_checkpoint()
        assert cp.iteration == 3
    
    def test_commit_iteration(self, git_state, temp_git_repo):
        """Should commit iteration with structured message."""
        workspace, ralph_dir = temp_git_repo
        
        # Create a file to commit
        (ralph_dir / "test.txt").write_text("test")
        
        result = git_state.commit_iteration(42, "oh-a1b2c", "Task completed")
        assert result is True
        
        # Check commit message
        log = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=workspace, capture_output=True, text=True
        )
        assert "[Ralph:Iter:42]" in log.stdout
        assert "oh-a1b2c" in log.stdout
    
    def test_learning_add_get(self, git_state, temp_git_repo):
        """Should add and retrieve learnings."""
        workspace, _ = temp_git_repo
        
        # Need a commit to attach notes to
        (workspace / "file.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
        subprocess.run(["git", "commit", "-m", "test"], cwd=workspace, capture_output=True)
        
        git_state.add_learning("Use pytest for testing")
        learnings = git_state.get_learnings()
        
        assert len(learnings) >= 1
        assert any("pytest" in l for l in learnings)
    
    def test_handoff_write_read_clear(self, git_state, temp_git_repo):
        """Should write, read, and clear handoff."""
        workspace, _ = temp_git_repo
        
        git_state.write_handoff("oh-a1b2c", "Continue from here", {"key": "value"})
        
        handoff = git_state.read_handoff()
        assert handoff is not None
        assert handoff["task_id"] == "oh-a1b2c"
        assert handoff["message"] == "Continue from here"
        assert handoff["context"]["key"] == "value"
        
        git_state.clear_handoff()
        assert git_state.read_handoff() is None
    
    def test_state_summary(self, git_state):
        """Should return state summary."""
        git_state.set_iteration(10)
        git_state.set_current_task("oh-test")
        git_state.set_status("running")
        
        summary = git_state.get_state_summary()
        
        assert summary["iteration"] == 10
        assert summary["task"] == "oh-test"
        assert summary["status"] == "running"
    
    def test_reset_state(self, git_state):
        """Should reset all state."""
        git_state.set_iteration(100)
        git_state.set_current_task("oh-task")
        git_state.set_status("running")
        
        git_state.reset_state()
        
        assert git_state.get_iteration() == 0
        assert git_state.get_current_task() == ""
        assert git_state.get_status() == "stopped"


# =============================================================================
# TASK MANAGER TESTS
# =============================================================================

class TestTaskManager:
    def test_create_task(self, task_manager):
        """Should create task with generated ID."""
        task = task_manager.create_task("Test task", "Description")
        
        assert task.id.startswith("ts-")
        assert task.title == "Test task"
        assert task.description == "Description"
        assert task.status == "pending"
    
    def test_get_task(self, task_manager):
        """Should retrieve task by ID."""
        task = task_manager.create_task("Test")
        retrieved = task_manager.get_task(task.id)
        
        assert retrieved is not None
        assert retrieved.id == task.id
        assert retrieved.title == "Test"
    
    def test_update_task(self, task_manager):
        """Should update task fields."""
        task = task_manager.create_task("Original")
        task_manager.update_task(task.id, title="Updated", status="active")
        
        updated = task_manager.get_task(task.id)
        assert updated.title == "Updated"
        assert updated.status == "active"
    
    def test_set_task_status(self, task_manager):
        """Should set task status."""
        task = task_manager.create_task("Test")
        task_manager.set_task_status(task.id, "done")
        
        assert task_manager.get_task(task.id).status == "done"
    
    def test_get_active_task(self, task_manager):
        """Should get currently active task."""
        task1 = task_manager.create_task("Task 1")
        task2 = task_manager.create_task("Task 2")
        
        task_manager.set_active_task(task2.id)
        
        active = task_manager.get_active_task()
        assert active is not None
        assert active.id == task2.id
    
    def test_set_active_deactivates_others(self, task_manager):
        """Setting active should deactivate other tasks."""
        task1 = task_manager.create_task("Task 1")
        task2 = task_manager.create_task("Task 2")
        
        task_manager.set_active_task(task1.id)
        task_manager.set_active_task(task2.id)
        
        t1 = task_manager.get_task(task1.id)
        t2 = task_manager.get_task(task2.id)
        
        assert t1.status == "pending"
        assert t2.status == "active"
    
    def test_get_next_task_no_deps(self, task_manager):
        """Should get next pending task."""
        task_manager.create_task("Task 1")
        task_manager.create_task("Task 2")
        
        next_task = task_manager.get_next_task()
        assert next_task is not None
        assert next_task.status == "pending"
    
    def test_get_next_task_with_deps(self, task_manager):
        """Should respect dependencies."""
        task1 = task_manager.create_task("Task 1")
        task2 = task_manager.create_task("Task 2", depends=[task1.id])
        
        # Task 2 should not be returned (dependency not done)
        next_task = task_manager.get_next_task()
        assert next_task.id == task1.id
        
        # Complete task 1
        task_manager.set_task_status(task1.id, "done")
        
        # Now task 2 should be available
        next_task = task_manager.get_next_task()
        assert next_task.id == task2.id
    
    def test_get_progress(self, task_manager):
        """Should return progress counts."""
        task_manager.create_task("Task 1")
        task_manager.create_task("Task 2")
        task3 = task_manager.create_task("Task 3")
        task_manager.set_task_status(task3.id, "done")
        
        done, total = task_manager.get_progress()
        assert done == 1
        assert total == 3
    
    def test_delete_task(self, task_manager):
        """Should delete task."""
        task = task_manager.create_task("To delete")
        task_id = task.id
        
        result = task_manager.delete_task(task_id)
        assert result is True
        assert task_manager.get_task(task_id) is None
    
    def test_to_prompt_format(self, task_manager):
        """Should format tasks for LLM prompt."""
        task_manager.create_task("Task 1", "Description 1")
        task2 = task_manager.create_task("Task 2")
        task_manager.set_task_status(task2.id, "done")
        
        prompt = task_manager.to_prompt_format()
        
        assert "## Tasks" in prompt
        assert "Task 1" in prompt
        assert "Task 2" in prompt
        assert "⏳" in prompt  # Pending
        assert "✅" in prompt  # Done
    
    def test_persistence(self, temp_git_repo):
        """Tasks should persist across instances."""
        workspace, ralph_dir = temp_git_repo
        
        # Create tasks
        tm1 = TaskManager(workspace, "ts", ralph_dir)
        task = tm1.create_task("Persistent task")
        task_id = task.id
        
        # New instance should see tasks (from git blob)
        tm2 = TaskManager(workspace, "ts", ralph_dir)
        loaded = tm2.get_task(task_id)
        
        assert loaded is not None
        assert loaded.title == "Persistent task"


# =============================================================================
# FORMULA TESTS
# =============================================================================

class TestFormula:
    def test_formula_from_toml(self):
        """Should parse formula from TOML."""
        toml_content = '''
description = "Test formula"
formula = "test"
version = 1

[vars.name]
description = "Name variable"
required = true

[[steps]]
id = "step1"
title = "First step"
description = "Do {{name}}"

[[steps]]
id = "step2"
title = "Second step"
description = "After first"
needs = ["step1"]
'''
        formula = Formula.from_toml(toml_content)
        
        assert formula.name == "test"
        assert formula.description == "Test formula"
        assert len(formula.steps) == 2
        assert formula.steps[0].id == "step1"
        assert formula.steps[1].needs == ["step1"]
        assert "name" in formula.vars


class TestFormulaManager:
    def test_create_builtin_formulas(self, formula_manager):
        """Should create built-in formulas."""
        formula_manager.create_builtin_formulas()
        
        formulas = formula_manager.list_formulas()
        assert "bugfix" in formulas
        assert "feature" in formulas
        assert "refactor" in formulas
    
    def test_get_formula(self, formula_manager):
        """Should get formula by name."""
        formula_manager.create_builtin_formulas()
        
        bugfix = formula_manager.get_formula("bugfix")
        assert bugfix is not None
        assert bugfix.name == "bugfix"
        assert len(bugfix.steps) >= 2
    
    def test_cook_formula(self, formula_manager, task_manager):
        """Should create tasks from formula."""
        formula_manager.create_builtin_formulas()
        
        tasks = formula_manager.cook_formula(
            "bugfix",
            task_manager,
            vars={"bug_description": "Button doesn't work"}
        )
        
        assert len(tasks) >= 2
        assert any("reproduce" in t.title.lower() for t in tasks)
        assert any("fix" in t.title.lower() for t in tasks)
    
    def test_cook_formula_dependencies(self, formula_manager, task_manager):
        """Formula steps should create task dependencies."""
        formula_manager.create_builtin_formulas()
        
        tasks = formula_manager.cook_formula(
            "bugfix",
            task_manager,
            vars={"bug_description": "Test"}
        )
        
        # Find the fix task and check it depends on reproduce
        task_dict = {t.title.lower(): t for t in tasks}
        
        # At least one task should have dependencies
        has_deps = any(len(t.depends) > 0 for t in tasks)
        assert has_deps
    
    def test_cook_formula_missing_var(self, formula_manager, task_manager):
        """Should raise error for missing required variable."""
        formula_manager.create_builtin_formulas()
        
        with pytest.raises(ValueError, match="Required variable"):
            formula_manager.cook_formula("bugfix", task_manager, vars={})


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    def test_full_workflow(self, temp_git_repo):
        """Test complete workflow: create tasks, track progress, commit."""
        workspace, ralph_dir = temp_git_repo
        
        git_state = GitStateManager(workspace)
        task_manager = TaskManager(workspace, "test", ralph_dir)
        formula_manager = FormulaManager(workspace, ralph_dir)
        
        # Create formulas
        formula_manager.create_builtin_formulas()
        
        # Cook a formula
        tasks = formula_manager.cook_formula(
            "bugfix",
            task_manager,
            vars={"bug_description": "Login fails"}
        )
        
        # Start work
        git_state.set_status("running")
        task = task_manager.get_next_task()
        git_state.set_current_task(task.id)
        task_manager.set_active_task(task.id)
        
        # Simulate iteration
        git_state.set_iteration(1)
        
        # Add learning
        (workspace / "fix.py").write_text("# fix")
        subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
        subprocess.run(["git", "commit", "-m", "fix"], cwd=workspace, capture_output=True)
        git_state.add_learning("Always check null before accessing")
        
        # Complete task
        task_manager.set_task_status(task.id, "done")
        git_state.commit_iteration(1, task.id, "Completed reproduce step")
        git_state.save_checkpoint(1, task.id, "completed")
        
        # Verify state
        assert git_state.get_iteration() == 1
        done, total = task_manager.get_progress()
        assert done == 1
        
        cp = git_state.load_checkpoint()
        assert cp.status == "completed"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
