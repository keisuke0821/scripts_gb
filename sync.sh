#!/bin/bash

content_dir="/home/keisuke/git_script"

cd "$content_dir"
git add .
git commit -m "Commit at $(date "+%Y-%m-%d %T")" || true
git pull origin main --rebase
git push origin main
