model=softmax
checkpoint=149

echo $model
echo simmim
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 run_pretrain.py \
    --model $model \
    --seq_len 256 \
    >> ../models/seq_len-256/$model/logs/pretrain_output.txt 2>&1
    
for task in cifar10 cifar100 tiny-imagenet-200; do
    echo $model $task
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python3 run_ic.py \
        --model $model \
        --seq_len 256 \
        --batch_size 512 \
        --lr 0.0005 \
        --epoch 100 \
        --task $task \
        --checkpoint $checkpoint \
        >> ../models/seq_len-256/$model/logs/ic-$task-$checkpoint-0.0005-512-100.txt 2>&1
done
