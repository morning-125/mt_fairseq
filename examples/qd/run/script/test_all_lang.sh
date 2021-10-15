!/bin/bash
train_flag=false
keep_dir=true

gpu_device=1

data_dir=/data1/qd/gao/data-bin/ted_8_related
save_dir=/data1/qd/save_model/nmt/base_model

echo "test_progress_begin"

arr=(aze tur bel rus glg por slk ces)
for tar in ${arr[@]};do
echo $tar
CUDA_VISIBLE_DEVICES=$gpu_device fairseq-generate  $data_dir \
    -s eng -t $tar \
    --task multilingual_translation \
    --gen-subset 'test' \
    --path $save_dir/checkpoint_best.pt \
    --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
    --scoring sacrebleu \
    --remove-bpe 'sentencepiece' \
    --lenpen 1.0 \
    --beam 5  \
    --decoder-langtok \
    --max-tokens 4096 --quiet 
done
