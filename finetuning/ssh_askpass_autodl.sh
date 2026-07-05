#!/usr/bin/env bash
if [[ -z "${AUTODL_SSH_PASSWORD:-}" ]]; then
  echo "AUTODL_SSH_PASSWORD is not set" >&2
  exit 1
fi

printf '%s\n' "$AUTODL_SSH_PASSWORD"
