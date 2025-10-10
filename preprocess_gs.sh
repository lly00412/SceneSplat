#!/bin/bash

#scenes=( GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH )
#scenes=( GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH)

#for selected_scene in ${scenes[@]}
#do
#
#python scripts/preprocess_gs_v2.py \
#       --input /mnt/Data2/liyan/ActiveSGM/results/MP3D/${selected_scene}/ActiveSem/run_0/splatam/final/params.npz \
#       --output /mnt/Data3/liyan/preprocess/MP3D/ActiveSGM/${selected_scene}/
#
#done

scenes=(office0 office1 office2 office3 office4 room0 room1 room2)

for selected_scene in ${scenes[@]}
do

python scripts/preprocess_gs_v2.py \
       --input /mnt/Data2/liyan/ActiveSGM/results/Replica/${selected_scene}/ActiveSem/run_0/splatam/final/params.npz \
       --output /mnt/Data3/liyan/preprocess/Replica/ActiveSGM/${selected_scene}/

done
