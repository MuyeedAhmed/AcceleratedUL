#!/bin/bash

algo="$1"
system="$2"

python ACE_Run_Init.py "$algo" "$system"

declare -a file_array

while IFS= read -r line
do
  file_array+=("$line")
done < "ace_master_files.txt"

rm ace_master_files.txt

for file in "${file_array[@]}"; do
  echo "$file"
  python ACE_Run_File.py "$algo" "$system" "$file"
done

