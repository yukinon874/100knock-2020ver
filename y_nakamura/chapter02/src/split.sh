#!/bin/zsh

file_path=$argv[1]
n=$argv[2]

num_line=`cat $file_path | wc -l`
l_num=$((($num_line + $n - 1) / $n))
#file_name=$file_path"splited_file"$file_name"-"

split -l $l_num $file_path ./work/split/splited_popular-names-
