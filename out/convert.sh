#!/bin/bash

mkdir -p trash/

while :; do
    sleep 2
    for f in *.01.png; do
        [ ! -f $f ] && break
        num=$(echo "$f" | sed 's/\..*//g')
        echo $f $num
        convert +append $num.*.png row.$num.png
        mv $num.*.png trash/
    done
    convert -append row.*.png all.png
done
