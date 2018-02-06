export MODEL=$2
find ./$1/ -name "*.jpg" -exec sh -c 'exif "{}" 2>/dev/null | grep -l --label="{}" "${MODEL}$"' \; | xargs -I % sh -c 'identify -verbose "%" | egrep  -l --label="%" "Quality: (100|9[4-9])"' > $1_jpgs 2> >(egrep -o "\./.+/.+\.jpg" > $1_bad)
