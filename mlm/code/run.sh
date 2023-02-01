model=softmax
checkpoint=189

echo $model, mlm pre-train
python3 run_pretrain.py \
    --model $model \
    --seq_len 512 \
    >> ../models/seq_len-512/$model/logs/pretrain_output.txt 2>&1

echo $model, mlm validate, $checkpoint
python3 run_validate.py \
    --model $model \
    --checkpoint $checkpoint \
    >> ../models/seq_len-512/$model/logs/valid_output.txt 2>&1

for task in sst2 qnli mnli qqp; do
    echo $model $task
    CUDA_VISIBLE_DEVICES=0 \
    python3 run_glue.py \
        --model $model \
        --batch_size 32 \
        --lr 2e-5 \
        --epoch 5 \
        --task $task \
        --checkpoint $checkpoint \
        --inst_per_gpu 32 \
        --seq_len 512 \
        >> ../models/seq_len-512/$model/logs/$task-$checkpoint-2e-05-32-5.txt 2>&1
done
