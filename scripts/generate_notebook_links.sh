#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

users=($( git branch -r | grep -v master | while read remote; do echo "${remote:9}"; done ))
base="https://github.com/cl-tohoku/100knock-2017/blob"

for i in $(seq -f "%02g" 1 10)
do
  echo "### Chapter ${i}"
  echo ""

  for user in "${users[@]}"
  do
    link="${base}/${user}/${user}/chapter${i}/chapter${i}.ipynb"
    echo "- [${user}](${link})"
  done

  echo ""
done
