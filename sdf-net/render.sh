#!/bin/bash

shape_list=(
#thai
armadillo
#bunny
)

lod_list=(
1
)

path=/home/vanveen/nglod/sdf-net/_results
nlod=5

for shape in "${shape_list[@]}"; do
for lod in "${lod_list[@]}"; do
python3.8 app/sdf_renderer.py \
	--net OctreeSDF \
	--num-lods $nlod \
	--pretrained $path/models/nlod$nlod/$shape.pth \
	--img-dir $path/render_app/imgs_lod$lod \
	--render-res 2560 1440 \
	--shading-mode matcap \
	--exr \
	--r360 \
	--lod $lod \
	--gpu 7
done
done
