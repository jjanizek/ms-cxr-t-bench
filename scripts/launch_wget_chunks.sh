#!/bin/bash
# Launch 8 parallel wget -i sessions in a tmux window.
# Each session downloads ~8900 URLs from its chunk file using persistent connections.
# wget -i reuses TCP connections between files from the same host — much faster
# than spawning a new subprocess per file.
#
# Usage: bash scripts/launch_wget_chunks.sh
# Monitor: tmux attach -t imgdl  (then Ctrl-B + number to switch panes)

read -sp "PhysioNet password for jjanizek: " PW
echo

CHUNKS_DIR="data/imagenome_pairs/url_chunks"
DEST="/data/imagenome_images"
USER="jjanizek"

mkdir -p "$DEST"

# Create a fresh tmux session with 8 panes
tmux new-session -d -s imgdl -x 220 -y 50

for i in 0 1 2 3 4 5 6 7; do
    CHUNK="$CHUNKS_DIR/chunk_${i}.txt"
    CMD="wget --user='$USER' --password='$PW' -nd -nc -q --show-progress -P '$DEST' -i '$CHUNK'; echo 'CHUNK $i DONE'"
    if [ "$i" -eq 0 ]; then
        tmux send-keys -t imgdl "$CMD" Enter
    else
        tmux split-window -t imgdl "$CMD"
        tmux select-layout -t imgdl tiled
    fi
done

echo "8 wget sessions started in tmux session 'imgdl'."
echo "Attach with: tmux attach -t imgdl"
echo "Progress: watch -n 30 \"ls /data/imagenome_images/*.jpg | wc -l\""
