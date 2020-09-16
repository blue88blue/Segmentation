#!/bin/bash
# program:
#	wait the python stop and run train.py
# History:
#2020/7/16    JiahuanSong      first release

status=$(nvidia-smi | grep python | wc -L)
echo $status

until [[ $status -eq 0 ]]
do
 	sleep 100
	status=$(nvidia-smi | grep python | wc -L)
	echo $status
done

echo training start...

python train.py