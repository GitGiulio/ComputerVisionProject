#!/bin/bash
# ============================================================
#  GPU Monitor — multi-user Telegram bot with commands
# ============================================================

# --- Configuration ---
TOKEN_FILE="./.bot_token"
if [[ ! -f "$TOKEN_FILE" ]]; then
    echo "ERROR: Bot token file not found at $TOKEN_FILE" >&2
    exit 1
fi
TELEGRAM_BOT_TOKEN="$(cat "$TOKEN_FILE")"
GPU_THRESHOLD=9                          # % utilisation considered "free"
CHECK_INTERVAL=600                        # seconds between checks (600 = 10 min)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="./gpu_monitor.log"
CHAT_IDS_FILE="./chat_ids.txt"       # one chat_id per line
STATE_FILE="./user_state.json"       # tracks pause/resume per user
LAST_UPDATE_FILE="./last_update_id"  # tracks Telegram polling offset

# ============================================================
#  Helpers
# ============================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Send a Telegram message to a specific chat_id
send_telegram() {
    local chat_id="$1"
    local message="$2"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${chat_id}" \
        --data-urlencode "text=${message}" \
        --data-urlencode "parse_mode=Markdown" \
        > /dev/null
}

# Send a message to all active (non-paused) users
broadcast() {
    local message="$1"
    while IFS= read -r chat_id; do
        [[ -z "$chat_id" || "$chat_id" == \#* ]] && continue
        if ! is_paused "$chat_id"; then
            send_telegram "$chat_id" "$message"
            log "Notified chat_id=${chat_id}"
        else
            log "Skipped chat_id=${chat_id} (paused)"
        fi
    done < "$CHAT_IDS_FILE"
}

# ============================================================
#  State management (JSON via Python — available on clusters)
# ============================================================

# is_paused <chat_id>  → returns 0 (true) if user is currently paused
is_paused() {
    local chat_id="$1"
    python3 - "$chat_id" "$STATE_FILE" <<'EOF'
import sys, json, time
chat_id = sys.argv[1]
state_file = sys.argv[2]
try:
    with open(state_file) as f:
        state = json.load(f)
except Exception:
    sys.exit(1)  # not paused

user = state.get(chat_id, {})
if not user.get("paused", False):
    sys.exit(1)  # not paused

resume_at = user.get("resume_at")
if resume_at and time.time() >= resume_at:
    sys.exit(1)  # pause expired -> not paused

sys.exit(0)  # paused
EOF
}

# set_state <chat_id> <paused: true|false> [resume_at_epoch]
set_state() {
    local chat_id="$1"
    local paused="$2"
    local resume_at="${3:-null}"
    python3 - "$chat_id" "$paused" "$resume_at" "$STATE_FILE" <<'EOF'
import sys, json
chat_id, paused_str, resume_at_str, state_file = sys.argv[1:]
try:
    with open(state_file) as f:
        state = json.load(f)
except Exception:
    state = {}

paused = paused_str == "true"
resume_at = float(resume_at_str) if resume_at_str != "null" else None
state[chat_id] = {"paused": paused, "resume_at": resume_at}

with open(state_file, "w") as f:
    json.dump(state, f, indent=2)
EOF
}

# Expire any timed pauses (called each cycle so resume is automatic)
expire_pauses() {
    python3 - "$STATE_FILE" <<'EOF'
import sys, json, time
state_file = sys.argv[1]
try:
    with open(state_file) as f:
        state = json.load(f)
except Exception:
    sys.exit(0)

changed = False
for chat_id, data in state.items():
    if data.get("paused") and data.get("resume_at") and time.time() >= data["resume_at"]:
        data["paused"] = False
        data["resume_at"] = None
        changed = True
        print(f"AUTO_RESUME:{chat_id}")   # picked up by caller

if changed:
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
EOF
}

# ============================================================
#  Telegram command polling
# ============================================================

get_last_update_id() {
    if [[ -f "$LAST_UPDATE_FILE" ]]; then
        cat "$LAST_UPDATE_FILE"
    else
        echo "0"
    fi
}

save_last_update_id() {
    echo "$1" > "$LAST_UPDATE_FILE"
}

# Fetch and process any pending Telegram commands
process_commands() {
    local offset
    offset=$(( $(get_last_update_id) + 1 ))

    local response
    response=$(curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates?offset=${offset}&timeout=2")

    python3 - "$response" <<'EOF'
import sys, json
try:
    data = json.loads(sys.argv[1])
    updates = data.get("result", [])
    for u in updates:
        update_id = u.get("update_id", 0)
        msg = u.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = msg.get("text", "").strip()
        first_name = msg.get("from", {}).get("first_name", "User")
        if chat_id and text.startswith("/"):
            print(f"{update_id}\t{chat_id}\t{first_name}\t{text}")
        elif chat_id:
            print(f"{update_id}\t\t\t")
except Exception:
    pass
EOF
}

handle_commands() {
    local raw
    raw=$(process_commands)
    [[ -z "$raw" ]] && return

    local last_id=0

    while IFS=$'\t' read -r update_id chat_id first_name text; do
        [[ -z "$update_id" ]] && continue
        last_id="$update_id"
        [[ -z "$chat_id" || -z "$text" ]] && continue

        log "Command from chat_id=${chat_id} (${first_name}): ${text}"

        # Check this chat_id is registered
        if ! grep -qx "$chat_id" "$CHAT_IDS_FILE" 2>/dev/null; then
            send_telegram "$chat_id" "⛔ Your chat ID (\`${chat_id}\`) is not registered. Ask the admin to add it to \`chat_ids.txt\`."
            continue
        fi

        case "$text" in
            /pause)
                set_state "$chat_id" "true" "null"
                send_telegram "$chat_id" "⏸️ Notifications paused indefinitely. Send /resume to re-enable."
                log "Paused chat_id=${chat_id}"
                ;;

            /pause\ *h|/pause\ *H)
                hours=$(echo "$text" | grep -oP '\d+(?=[hH])')
                if [[ -z "$hours" || "$hours" -le 0 ]]; then
                    send_telegram "$chat_id" "❌ Usage: \`/pause Nh\` where N is the number of hours. Example: \`/pause 3h\`"
                else
                    resume_epoch=$(( $(date +%s) + hours * 3600 ))
                    resume_time=$(date -d "@${resume_epoch}" '+%Y-%m-%d %H:%M')
                    set_state "$chat_id" "true" "$resume_epoch"
                    send_telegram "$chat_id" "⏸️ Notifications paused for ${hours}h. Will auto-resume at ${resume_time}."
                    log "Paused chat_id=${chat_id} for ${hours}h (until ${resume_time})"
                fi
                ;;

            /resume)
                set_state "$chat_id" "false" "null"
                send_telegram "$chat_id" "▶️ Notifications resumed! You'll be alerted when a GPU is free."
                log "Resumed chat_id=${chat_id}"
                ;;

            /status)
                gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
                    --format=csv,noheader,nounits 2>/dev/null | \
                    awk -F',' '{printf "GPU %s: %s%% util | %s/%s MiB\n", $1, $2, $3, $4}')
                if is_paused "$chat_id"; then
                    pause_status="⏸️ You are currently *paused*."
                else
                    pause_status="▶️ You are currently *active*."
                fi
                send_telegram "$chat_id" "📊 *Current GPU status:*
\`\`\`
${gpu_info}
\`\`\`
${pause_status}"
                ;;

            /help)
                send_telegram "$chat_id" "🤖 *GPU Monitor Bot*

Available commands:
/status — show current GPU utilisation
/pause — stop notifications indefinitely
/pause Nh — pause for N hours (e.g. \`/pause 3h\`)
/resume — re-enable notifications
/help — show this message"
                ;;

            *)
                send_telegram "$chat_id" "❓ Unknown command. Send /help for available commands."
                ;;
        esac
    done <<< "$raw"

    [[ "$last_id" -gt 0 ]] && save_last_update_id "$last_id"
}

# ============================================================
#  Initialisation
# ============================================================

[[ ! -f "$CHAT_IDS_FILE" ]]    && touch "$CHAT_IDS_FILE"
[[ ! -f "$STATE_FILE" ]]       && echo "{}" > "$STATE_FILE"
[[ ! -f "$LAST_UPDATE_FILE" ]] && echo "0" > "$LAST_UPDATE_FILE"

log "GPU monitor started (threshold: <${GPU_THRESHOLD}%, interval: ${CHECK_INTERVAL}s)"

while IFS= read -r chat_id; do
    [[ -z "$chat_id" || "$chat_id" == \#* ]] && continue
    send_telegram "$chat_id" "🟢 GPU monitor started on \`$(hostname)\`. Checking every $((CHECK_INTERVAL/60)) min. Send /help for commands."
done < "$CHAT_IDS_FILE"

# ============================================================
#  Main loop
# ============================================================

while true; do
    # 1. Process any pending Telegram commands
    handle_commands

    # 2. Auto-expire timed pauses
    while IFS= read -r line; do
        if [[ "$line" == AUTO_RESUME:* ]]; then
            chat_id="${line#AUTO_RESUME:}"
            log "Auto-resumed chat_id=${chat_id}"
            send_telegram "$chat_id" "▶️ Your timed pause has expired — notifications resumed!"
        fi
    done < <(expire_pauses)

    # 3. Check GPU utilisation
    mapfile -t GPU_DATA < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')

    if [[ ${#GPU_DATA[@]} -eq 0 ]]; then
        log "ERROR: nvidia-smi returned no data."
        sleep "$CHECK_INTERVAL"
        continue
    fi

    free_gpus=()
    status_lines=()

    for entry in "${GPU_DATA[@]}"; do
        idx="${entry%%,*}"
        util="${entry##*,}"
        status_lines+=("GPU ${idx}: ${util}%")
        if [[ "$util" -lt "$GPU_THRESHOLD" ]]; then
            free_gpus+=("GPU ${idx} (${util}%)")
        fi
    done

    status_summary=$(IFS=$'\n'; echo "${status_lines[*]}")
    log "Status — ${status_summary//$'\n'/' | '}"

    # 4. Broadcast alert if any GPU is free
    if [[ ${#free_gpus[@]} -gt 0 ]]; then
        free_list=$(IFS=", "; echo "${free_gpus[*]}")
        message="🚀 *GPU available on \`$(hostname)\`!*

Free: ${free_list}

Current utilisation:
\`\`\`
${status_summary}
\`\`\`
Connect now and grab it! ⚡"

        log "ALERT: Free GPU(s) — ${free_list}. Broadcasting."
        broadcast "$message"
    fi

    sleep "$CHECK_INTERVAL"
done