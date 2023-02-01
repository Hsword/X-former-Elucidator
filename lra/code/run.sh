pooling=cls
seed=0
task=text
model=softmax

echo $model
modelshort=${model%%-*}
echo $task
echo $pooling

CUDA_VISIBLE_DEVICES=0 \
python3 run_task.py \
    --model $model \
    --task $task \
    --cls_head $pooling \
    --seed $seed \
    >> ../models/$pooling/$modelshort/logs/$task"_"$seed.txt 2>&1
