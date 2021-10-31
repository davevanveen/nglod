shape=lucy

python3.8 app/main.py \
    --net OctreeSDF \
    --num-lods 4 \
    --dataset-path /home/vanveen/nglod/data/$shape.obj \
    --epoch 250 \
    --gpu 5 \
    --exp-name $shape
