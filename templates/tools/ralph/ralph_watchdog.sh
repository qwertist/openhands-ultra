#!/bin/bash
################################################################################
# Ralph Watchdog - Auto-restart daemon if it dies
# 
# Run this script with: nohup /workspace/.ralph/ralph_watchdog.sh &
# Or via cron: */1 * * * * /workspace/.ralph/ralph_watchdog.sh
################################################################################

RALPH_DIR="/workspace/.ralph"
DAEMON_SCRIPT="$RALPH_DIR/ralph_daemon.py"
PID_FILE="$RALPH_DIR/ralph_daemon.pid"
HEARTBEAT_FILE="$RALPH_DIR/heartbeat"
LOG_FILE="$RALPH_DIR/watchdog.log"
CONFIG_FILE="$RALPH_DIR/config.json"

# Max age of heartbeat in seconds before considering daemon dead
MAX_HEARTBEAT_AGE=120

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

get_config_status() {
    if [ -f "$CONFIG_FILE" ]; then
        python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('status', 'unknown'))" 2>/dev/null
    else
        echo "no_config"
    fi
}

is_daemon_running() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0  # Running
        fi
    fi
    
    # Check by process name
    if pgrep -f "ralph_daemon.py" > /dev/null 2>&1; then
        return 0  # Running
    fi
    
    return 1  # Not running
}

is_heartbeat_stale() {
    if [ ! -f "$HEARTBEAT_FILE" ]; then
        return 0  # No heartbeat = stale
    fi
    
    # CRITICAL FIX: Handle clock skew between container and host
    # by using absolute value of age difference
    heartbeat_time=$(cat "$HEARTBEAT_FILE" 2>/dev/null)
    current_time=$(date +%s)
    age=$((current_time - heartbeat_time))
    
    # Handle negative age (clock skew) by using absolute value
    if [ "$age" -lt 0 ]; then
        age=$((-age))
    fi
    
    if [ "$age" -gt "$MAX_HEARTBEAT_AGE" ]; then
        return 0  # Stale
    fi
    
    return 1  # Fresh
}

start_daemon() {
    log "Starting Ralph daemon..."
    cd /workspace
    nohup python3 "$DAEMON_SCRIPT" >> "$RALPH_DIR/ralph_daemon.log" 2>&1 &
    new_pid=$!
    echo "$new_pid" > "$PID_FILE"
    log "Daemon started with PID $new_pid"
}

kill_daemon() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$pid" ]; then
            log "Killing daemon PID $pid"
            kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$PID_FILE"
    fi
    
    # Kill any orphaned processes
    pkill -9 -f "ralph_daemon.py" 2>/dev/null
}

# Main logic
main() {
    # Check if daemon script exists
    if [ ! -f "$DAEMON_SCRIPT" ]; then
        log "ERROR: Daemon script not found: $DAEMON_SCRIPT"
        exit 1
    fi
    
    # Get current status from config
    status=$(get_config_status)
    
    # Only manage daemon if status is 'running'
    if [ "$status" != "running" ]; then
        # Not supposed to be running
        if is_daemon_running; then
            # Daemon running but status not 'running' - check if it should stop
            if [ "$status" = "stopped" ] || [ "$status" = "complete" ]; then
                log "Status is $status, stopping daemon"
                kill_daemon
            fi
        fi
        exit 0
    fi
    
    # Status is 'running' - ensure daemon is alive
    if ! is_daemon_running; then
        log "Daemon not running, starting..."
        start_daemon
        exit 0
    fi
    
    # Daemon running, check heartbeat
    if is_heartbeat_stale; then
        log "Heartbeat stale (>$MAX_HEARTBEAT_AGE s), restarting daemon..."
        kill_daemon
        sleep 2
        start_daemon
        exit 0
    fi
    
    # All good
    exit 0
}

main "$@"
