#!/bin/bash
git checkout master
git branch -r | grep -v '\->' | while read remote; do git branch --track "${remote#origin/}" "$remote"; done
git fetch --all
git branch -r | grep -v '\->' | while read remote; do git merge --ff $remote; done
