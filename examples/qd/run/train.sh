lang_pairs_str="eng-aze,eng-bel,eng-ces,eng-glg,eng-por,eng-rus,eng-slk,eng-tur"
databin_dir=/data1/qd/gao/data-bin/ted_8_related

fairseq-train ${databin_dir} \
  --lang-pairs "${lang_pairs_str}" \
  --arch multilingual_transformer_iwslt_de_en \
  --task multilingual_translation \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --share-encoders \
  --share-decoders \
  --decoder-langtok \
  --share-decoder-input-output-embed \
  --dropout 0.3 --attention-dropout 0.3 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --stop-min-lr 1e-9 --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --max-tokens 4096 --update-freq 1  \
  --lr 0.0015 \
  --clip-norm 1.0 \
  --seed 2 \
  --encoder-layers 6 \
  --decoder-layers 6 \