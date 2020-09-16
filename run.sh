#!/bin/bash
# program:
#	check the free GPUs and run train.py 
# History:
#2020/6/30    JiahuanSong      first release

gpu_id=$(python utils/get_gpu.py)
echo $gpu_id

until [[ $gpu_id -ge 0 ]]
do
 	sleep 3
	gpu_id=$(python utils/get_gpu.py)
	echo $gpu_id
done

CUDA_VISIBLE_DEVICES=$gpu_id python train.py


