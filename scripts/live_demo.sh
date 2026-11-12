#!/usr/bin/env bash
# The demonstration's live path: a new image and claim through every stage, to a local page.
#
#   PE_FAKE_NEWS_SERVER=user@host scripts/live_demo.sh --image PATH --claim "TEXT" [--date YYYY-MM-DD]
#   PE_FAKE_NEWS_SERVER=user@host scripts/live_demo.sh --requests FILE.jsonl     # {image, claim, date} per line
#
# Run from WSL in the repository. Paths are relative to it: the laptop's python is Windows'.
# The laptop does the gallery (faces, names), retrieval (with a date only: the as-of rule) and
# the verdicts. The server does the image model and the text model (extraction, prediction,
# the judge), each as one GPU job through ~/bin/bgrun.sh, refused while anything else holds the
# GPU, each refusing to run under any configuration but the evaluated one. Every ssh call is
# short; a job is waited for by polling its pid. The server must run the laptop's committed HEAD.
set -euo pipefail
: "${PE_FAKE_NEWS_SERVER:?set PE_FAKE_NEWS_SERVER to user@host (ssh key auth)}"
REMOTE=${PE_FAKE_NEWS_SERVER_ROOT:-Pe_Fake_News_Dec}   # the repository, relative to the server's home
PY=${PE_FAKE_NEWS_PYTHON:-./.venv/Scripts/python.exe}
LIVE=data/interim/consistency/c/live
cd "$(dirname "$0")/.."

q() { ssh -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
        -o ConnectTimeout=20 "$PE_FAKE_NEWS_SERVER" "$@"; }
files() { tr ',' '\n' <<<"$1" | sed "s#.*#$LIVE/&/$2#" | tr '\n' ' '; }

here=$(git -c core.filemode=false -c core.autocrlf=true rev-parse HEAD)
dirty=$(git -c core.filemode=false -c core.autocrlf=true status --porcelain -- scripts configs)
there=$(q "cat ~/$REMOTE/code_commit.txt")
if [ -n "$dirty" ] || [ "$here" != "$there" ]; then
  echo "REFUSED: the laptop runs ${here:0:7}${dirty:+ with uncommitted code}, the server ${there:0:7}: ship HEAD first" >&2
  exit 3
fi

gpu_job() {  # gpu_job NAME ARGS...: one consistency.py GPU job through bgrun, waited for
  local name=$1 state; shift
  q "n=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); \
     if [ \"\$n\" -ne 0 ]; then echo \"REFUSED: the GPU has \$n process(es)\"; exit 3; fi; \
     cd ~/$REMOTE && ~/bin/bgrun.sh $name ~/$REMOTE/.venv/bin/python scripts/consistency.py $* | head -n 2"
  while :; do
    sleep 15
    state=$(q "ps -p \$(cat ~/logs/$name.latest.pid) >/dev/null 2>&1 && echo RUN || echo DONE" 2>/dev/null || true)
    [ "$state" = "DONE" ] && break   # an ssh failure is "still running", never "done"
  done
  q "L=\$(readlink ~/logs/$name.latest.pid); grep -E 'REFUSED|Traceback|Error|^live|\\[live' ~/logs/\${L%.pid}.log | tail -n 20"
}

ens_job() {  # ens_job NAME ARGS...: one ensemble.py GPU job through bgrun, waited for
  local name=$1 state; shift
  q "n=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); \
     if [ \"\$n\" -ne 0 ]; then echo \"REFUSED: the GPU has \$n process(es)\"; exit 3; fi; \
     cd ~/$REMOTE && ~/bin/bgrun.sh $name ~/$REMOTE/.venv/bin/python scripts/ensemble.py $* | head -n 2"
  while :; do
    sleep 15
    state=$(q "ps -p \$(cat ~/logs/$name.latest.pid) >/dev/null 2>&1 && echo RUN || echo DONE" 2>/dev/null || true)
    [ "$state" = "DONE" ] && break   # an ssh failure is "still running", never "done"
  done
  q "L=\$(readlink ~/logs/$name.latest.pid); grep -E 'REFUSED|Traceback|Error|written|to do' ~/logs/\${L%.pid}.log | tail -n 5"
}

keys=$("$PY" scripts/consistency.py live prepare "$@" | tail -n 1 | tr -d '\r')
echo "requests: $keys"
tar czf - $(tr ',' '\n' <<<"$keys" | sed "s#^#$LIVE/#") | q "cd ~/$REMOTE && tar xzf -"
gpu_job live_vlm live-server vlm --keys "$keys"
q "cd ~/$REMOTE && tar czf - $(files "$keys" image_text.json)" | tar xzf -
"$PY" scripts/consistency.py live prompts --keys "$keys"
tar czf - $(files "$keys" llm_in.json) | q "cd ~/$REMOTE && tar xzf -"
gpu_job live_llm live-server llm --keys "$keys"
q "cd ~/$REMOTE && tar czf - $(files "$keys" llm_out.json)" | tar xzf -
"$PY" scripts/consistency.py live finish --keys "$keys"
# demo system v1.1: the five scorers on the server, then its flag and the page on the laptop
ENS=data/interim/ensemble/live
"$PY" scripts/ensemble.py live inputs --keys "$keys"
tar czf - $ENS/pairs_in.jsonl $ENS/pairs_in.meta.json $ENS/nli_in.jsonl $ENS/nli_in.meta.json $ENS/images.jsonl \
  | q "cd ~/$REMOTE && tar xzf -"
for s in openclip_h14 siglip_so400m blip_itm_large blip2_itm_g nli_deberta_large; do
  ens_job live_$s score --scorer $s --set live
done
q "cd ~/$REMOTE && tar czf - $ENS/scores" | tar xzf -
"$PY" scripts/ensemble.py live score --keys "$keys"
echo "keys: $keys"
