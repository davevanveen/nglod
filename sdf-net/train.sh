shape=bunny

python3.8 app/main.py \
    --net OctreeSDF \
    --num-lods 1 \
    --dataset-path ./gt/$shape.obj \
    --epoch 250 \
    --gpu 0 \
    --exp-name test #$shape
