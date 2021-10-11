!/bin/bash
train_flag=true
keep_dir=true

gpu_device=7 #0,1,2,3,4,5,6

data_dir=/data1/qd/gao/data-bin/ted_8_related
save_dir=/data1/qd/checkpoint

if [ $train_flag == true ]; then
    echo "train nmt model from lang $src to $tgt "

    if [ $keep_dir != true ]; then
        rm -rf $save_dir
    fi
    mkdir -p $save_dir

    CUDA_VISIBLE_DEVICES=$gpu_device fairseq-train $data_dir \
      --ddp-backend=legacy_ddp \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	  --task multilingual_translation \
	  --arch multilingual_transformer_iwslt_de_en \
      --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
      --share-encoders \
      --share-decoders \
      --decoder-langtok \
      --share-decoder-input-output-embed \
      --dropout 0.3 --attention-dropout 0.3 \
      --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
      --lr-scheduler inverse_sqrt --stop-min-lr 1e-9 --warmup-init-lr 1e-7 --warmup-updates 8000 \
      --max-tokens 3072 --update-freq 1  \
      --lr 0.0015 \
      --clip-norm 1.0 \
	  --seed 2 \
      --no-progress-bar \
      --fp16 \
      --log-interval 100 \
      --save-dir $save_dir | tee -a $save_dir/train.log \

else

    CUDA_VISIBLE_DEVICES=$gpu_device fairseq-generate  $data_dir \
        -s eng -t ces \
	    --user-dir examples/da_rdrop/base \
	    --task rdrop_multilingual_translation \
        --gen-subset 'test' \
        --path $save_dir/checkpoint_last.pt \
        --lang-pairs "eng-aze,eng-tur,eng-bel,eng-rus,eng-glg,eng-por,eng-slk,eng-ces" \
        --scoring sacrebleu \
        --remove-bpe 'sentencepiece' \
        --lenpen 1.0 \
        --beam 5  \
        --decoder-langtok \
        --max-tokens 4096 --quiet \


fi
