#!/usr/bin/env bash
set -u

HOST="${HOST:-connect.bjb1.seetacloud.com}"
PORT="${PORT:-48966}"
USER_NAME="${USER_NAME:-root}"
KEY_FILE="${KEY_FILE:-/media/data/zhangjingyi/.ssh/autodl_ed25519.pub}"
ASKPASS_SCRIPT="${ASKPASS_SCRIPT:-/media/data/zhangjingyi/ImAge/finetuning/ssh_askpass_autodl.sh}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
LOG_FILE="${LOG_FILE:-/media/data/zhangjingyi/ImAge/finetuning/install_autodl_key.log}"

log() {
  echo "[install-key] $(date '+%F %T') $*" | tee -a "$LOG_FILE"
}

if [[ ! -f "$KEY_FILE" ]]; then
  log "public key not found: $KEY_FILE"
  exit 1
fi

if [[ -z "${AUTODL_SSH_PASSWORD:-}" ]]; then
  log "AUTODL_SSH_PASSWORD is not set"
  exit 1
fi

chmod 700 /media/data/zhangjingyi/.ssh
chmod 600 /media/data/zhangjingyi/.ssh/config /media/data/zhangjingyi/.ssh/autodl_ed25519
chmod 644 "$KEY_FILE"
chmod +x "$ASKPASS_SCRIPT"

while true; do
  if timeout 5 bash -lc "</dev/tcp/$HOST/$PORT" 2>/dev/null; then
    log "SSH port is open, trying to install public key"
    export SSH_ASKPASS="$ASKPASS_SCRIPT"
    export SSH_ASKPASS_REQUIRE=force
    export DISPLAY="${DISPLAY:-dummy:0}"

    if setsid ssh-copy-id \
      -i "$KEY_FILE" \
      -o StrictHostKeyChecking=no \
      -p "$PORT" \
      "${USER_NAME}@${HOST}" </dev/null >>"$LOG_FILE" 2>&1; then
      log "public key installed successfully"
      exit 0
    fi

    log "ssh-copy-id failed, will retry"
  else
    log "SSH port not ready yet"
  fi

  sleep "$CHECK_INTERVAL"
done
