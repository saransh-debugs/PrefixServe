#!/usr/bin/env bash
# Dev / bench VM on GCP. Run from the repo root.
#
#   scripts/gcp.sh create      # L4 spot VM (default) or GPU=a100 for the A100
#                              # SNAPSHOT=name ZONE=z to restore a saved disk in another zone
#   scripts/gcp.sh sync        # copy the working tree up (no venvs, no build dirs)
#   scripts/gcp.sh run 'cmd'   # run a command in ~/PrefixServe on the VM
#   scripts/gcp.sh ssh | stop | start | delete | status
#
# Money safety:
# - Spot VMs, stopped (not deleted) when Google reclaims them.
# - The VM powers itself off after 30 min with an idle GPU and no SSH session.
set -euo pipefail

PROJECT="${PROJECT:-govrag-portfolio-fe8b}"
ZONE="${ZONE:-us-central1-a}"  # spot L4 stock moves between zones; see SNAPSHOT below
GPU="${GPU:-l4}"
case "$GPU" in
  # g2 / a2 machine types come with their GPU attached; no --accelerator flag.
  l4)   NAME="${NAME:-prefixserve-l4b}";   MACHINE="g2-standard-8"; PROV="SPOT" ;;
  a100) NAME="${NAME:-prefixserve-a100}"; MACHINE="a2-highgpu-1g"; PROV="STANDARD" ;;
  *) echo "GPU must be l4 or a100" >&2; exit 2 ;;
esac
G=(--project "$PROJECT" --zone "$ZONE")
REMOTE_DIR="PrefixServe"

# Powers off after 6 checks x 5 min with GPU util 0 and nobody logged in.
read -r -d '' IDLE_SCRIPT <<'EOS' || true
#!/bin/bash
cat > /usr/local/bin/idle-shutdown.sh <<'EOF'
#!/bin/bash
f=/var/run/idle-count
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
users=$(who | wc -l)
if [ "${util:-0}" -eq 0 ] && [ "$users" -eq 0 ]; then
  n=$(( $(cat $f 2>/dev/null || echo 0) + 1 )); echo $n > $f
  [ $n -ge 6 ] && shutdown -h now
else
  echo 0 > $f
fi
EOF
chmod +x /usr/local/bin/idle-shutdown.sh
echo '*/5 * * * * root /usr/local/bin/idle-shutdown.sh' > /etc/cron.d/idle-shutdown
EOS

cmd="${1:-}"; shift || true
case "$cmd" in
  create)
    startup=$(mktemp); trap 'rm -f "$startup"' EXIT
    printf '%s\n' "$IDLE_SCRIPT" > "$startup"
    # SNAPSHOT=name: boot from a snapshot of an earlier VM (keeps the venv
    # and downloaded models). Snapshots are global, so this is also how the
    # VM moves to another zone when one runs out of spot GPUs.
    if [ -n "${SNAPSHOT:-}" ]; then
      disk_src=(--source-snapshot="$SNAPSHOT")
    else
      disk_src=(--image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release)
    fi
    extra=()
    [ "$PROV" = "SPOT" ] && extra+=(--provisioning-model=SPOT --instance-termination-action=STOP)
    gcloud compute instances create "$NAME" "${G[@]}" \
      --machine-type="$MACHINE" \
      --maintenance-policy=TERMINATE \
      "${disk_src[@]}" \
      --boot-disk-size=200GB --boot-disk-type=pd-balanced \
      --metadata=install-nvidia-driver=True \
      --metadata-from-file=startup-script="$startup" \
      ${extra[@]+"${extra[@]}"}
    gcloud compute config-ssh --project "$PROJECT" >/dev/null
    echo "created $NAME. ssh host: $NAME.$ZONE.$PROJECT"
    ;;
  sync)
    gcloud compute config-ssh --project "$PROJECT" >/dev/null
    rsync -az --delete \
      --exclude .git --exclude '.venv*' --exclude build --exclude '_skbuild' \
      --exclude __pycache__ --exclude .pytest_cache --exclude .mypy_cache --exclude .hypothesis \
      --exclude '*.so' --exclude .DS_Store \
      --exclude PROMPTS.md --exclude INTERVIEW_PREP.md --exclude PLAN.md --exclude LEARN.md \
      ./ "$NAME.$ZONE.$PROJECT:$REMOTE_DIR/"
    ;;
  run)    gcloud compute ssh "$NAME" "${G[@]}" -- "cd $REMOTE_DIR && $*" ;;
  ssh)    gcloud compute ssh "$NAME" "${G[@]}" ;;
  start)  gcloud compute instances start "$NAME" "${G[@]}" ;;
  stop)   gcloud compute instances stop "$NAME" "${G[@]}" ;;
  delete) gcloud compute instances delete "$NAME" "${G[@]}" ;;
  status) gcloud compute instances describe "$NAME" "${G[@]}" --format="value(status)" ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
