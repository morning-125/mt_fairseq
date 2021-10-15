!/bin/bash
train_flag=false
keep_dir=true

gpu_device=0,1,2,3,4,5,6,7

data_dir=/data1/qd/gao/data-bin/ted_8_related

echo "test_progress_begin"

arr=(aze tur bel rus glg por slk ces)
for tar in ${arr[@]};do
save_dir=/data1/qd/save_model/nmt/O2O/$tar
    if [ $keep_dir != true ]; then
        rm -rf $save_dir
    fi
    mkdir -p $save_dir
    langp=eng-$tar
    CUDA_VISIBLE_DEVICES=$gpu_device fairseq-train $data_dir \
      --ddp-backend=legacy_ddp \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	  --task multilingual_translation \
	  --arch multilingual_transformer_iwslt_de_en \
      --lang-pairs $langp \
      --share-encoders \
      --share-decoders \
      --decoder-langtok \
      --share-decoder-input-output-embed \
      --dropout 0.3 --attention-dropout 0.3 \
      --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
      --lr-scheduler inverse_sqrt  --warmup-init-lr 1e-7 --warmup-updates 8000 \
      --max-tokens 3072 --update-freq 1  \
      --lr 0.0015 \
      --clip-norm 1.0 \
	  --seed 2 \
      --tensorboard-logdir ./tensorboard \
      --fp16 \
      --log-interval 100 \
      --save-dir $save_dir | tee -a $save_dir/train.log \

done
