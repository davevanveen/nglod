#!/bin/bash

shape_list=(
thai
armadillo
dragon
lucy
sphere
)

shape_list=(
sphere
)


lod_list=(
1
2
3
)

path=.
nlod=4

for shape in "${shape_list[@]}"; do
for lod in "${lod_list[@]}"; do
python3.8 app/render_3d.py \
	--net OctreeSDF \
	--num-lods $nlod \
    --out_file nglod_${shape}_${nlod}_${lod} \
    --shape $shape \
	--pretrained $path/models/nlod$nlod/$shape.pth \
	--img-dir $path/render_app/imgs_lod$lod \
	--render-res 2560 1440 \
	--shading-mode matcap \
	--exr \
	--lod $lod \
	--gpu 0 \
    --mrc
done
done
