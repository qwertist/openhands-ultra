---
name: tool-security
description: CRITICAL - Required security_risk parameter for ALL tool calls. Without this parameter your actions WILL FAIL.
always_active: true
triggers:
- the
- a
- is
- to
- and
---

# ⚠️ CRITICAL: REQUIRED PARAMETERS FOR ALL TOOL CALLS

**Your tool calls WILL FAIL without these required parameters.**

## EVERY tool call MUST include:

1. **`security_risk`** - REQUIRED, one of: `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"UNKNOWN"`
2. **`summary`** - REQUIRED, brief description (~10 words)

## Quick Reference

| Action Type | security_risk |
|-------------|---------------|
| Read files, list dirs, grep, find, cat | `"LOW"` |
| Write files, edit, create, install packages | `"MEDIUM"` |
| Execute remote scripts, system changes | `"HIGH"` |

## Correct Tool Call Format

### Terminal
```json
{
  "command": "ls -la",
  "security_risk": "LOW",
  "summary": "List files in current directory"
}
```

```json
{
  "command": "pip install pandas",
  "security_risk": "MEDIUM",
  "summary": "Install pandas package"
}
```

### File Editor
```json
{
  "command": "view",
  "path": "/workspace/file.py",
  "security_risk": "LOW",
  "summary": "View file contents"
}
```

```json
{
  "command": "str_replace",
  "path": "/workspace/file.py",
  "old_str": "old code",
  "new_str": "new code",
  "security_risk": "MEDIUM",
  "summary": "Update function implementation"
}
```

```json
{
  "command": "create",
  "path": "/workspace/new_file.py",
  "file_text": "# content",
  "security_risk": "MEDIUM",
  "summary": "Create new Python file"
}
```

### Task Tracker
```json
{
  "command": "view",
  "security_risk": "LOW",
  "summary": "View current tasks"
}
```

## Common Mistakes

❌ WRONG - missing security_risk:
```json
{"command": "ls -la"}
```

✅ CORRECT:
```json
{"command": "ls -la", "security_risk": "LOW", "summary": "List files"}
```

**REMEMBER: Always include `security_risk` and `summary` in EVERY tool call!**
