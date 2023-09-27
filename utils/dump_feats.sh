#!/bin/bash
. ./cmd.sh || exit 1;

nj=16     # number of parallel jobs in feature extraction

part="all" # data partition in LibriTTS

train_set="train_${part}" # name of training data directory
dev_set="dev_${part}"           # name of development data directory
eval_set="eval_${part}"         # name of evaluation data directory

stage=0
stop_stage=100

. parse_options.sh || exit 1;  # This allows you to pass command line arguments, e.g. --fmax 7600
set -eo pipefail

datadir=$PWD/data
featdir=$PWD/feats
vqdir=${featdir}/vqidx
ppedir=${featdir}/normed_ppe
fbankdir=${featdir}/normed_fbank
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Concatenating normed PPE, VQ-idx and normed FBANK features"
    for x in ${train_set} ${dev_set} ${eval_set} ; do
        dump_feat_dir=${featdir}/dump/${x}
        mkdir -p ${dump_feat_dir}

        split_scps=""
        for n in $(seq $nj); do
            split_scps="$split_scps ${ppedir}/${x}/feats.${n}.scp"
        done
        split_scp.pl ${ppedir}/${x}/feats.scp $split_scps

        ${train_cmd} JOB=1:${nj} exp/dump_feats/${x}/dump_feats.JOB.log \
        paste-feats.py --length-tolerance=2 \
                       scp:${ppedir}/${x}/feats.JOB.scp \
                       scp:${vqdir}/${x}/feats.scp \
                       scp:${fbankdir}/${x}/feats.scp \
                       ark,scp:${dump_feat_dir}/feats.JOB.ark,${dump_feat_dir}/feats.JOB.scp \
                       --compress True
        cat ${dump_feat_dir}/feats.*.scp | sort > ${datadir}/${x}/feats.scp
    done
fi
