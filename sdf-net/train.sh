shape=bunny

python3.8 app/main.py \
    --net OctreeSDF \
    --num-lods 5 \
    --dataset-path /home/vanveen/nglod/data/$shape.obj \
    --epoch 250 \
    --gpu 7 \
    --exp-name $shape
