# CTX-vec2wav, the Acoustic Context-Aware Vocoder

> This is the official implementation of **CTX-vec2wav** vocoder in the paper [UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding](https://arxiv.org/abs/2306.07547).

<img width="1187" alt="image-20230926140022539" src=asset/main.png>

## Environment Setup

This repo is tested on **python 3.9** on Linux. You can set up the environment with conda
```shell
# Install required packages
conda create -n ctxv2w python=3.9  # or any name you like
conda activate ctxv2w
pip install -r requirements.txt

# Then, set PATH and PYTHONPATH
source path.sh  # change the env name if you don't use "ctxv2w"
```
The scripts in `utils/` should be executable. You can run `chmod +x utils/*` to ensure this.

The following process will also need `bash` and `perl` commands in your Linux environment.


## Inference (Vocoding with acoustic context)
For utterances that are already registered in `data/`, the inference (VQ index + acoustic prompt) can be done by
```shell
bash run.sh --stage 3 --stop_stage 3
# You can specify the dataset to be constructed by "--eval_set $which_set", i.e. "--eval_set dev_all"
```
You can also create a subset can perform inference on it by
```shell
subset_data_dir.sh data/eval_all 200 data/eval_subset  # randomly select 200 utts from data/eval_all
bash run.sh --stage 3 --stop_stage 3 --eval_set "eval_subset"
```
The program loads the latest checkpoint in the experiment dir `exp/train_all_ctxv2w.v1/*pkl`.


💡Note: the stage 3 in `run.sh` automatically selects the prompt for each utterance by random (see `local/build_prompt_feat.py`).
You can customize this process and perform inference yourself:
1. Prepare a `feats.scp` that specifies each utterance (for inference) with its VQ index sequence in `(L, 2)` shape.
2. Run `feat-to-len.py scp:/path/to/feats.scp > /path/to/utt2num_frames`.
3. Prepare a `prompt.scp` that specifies each utterance with its acoustic (mel) prompt in `(L', 80)` shape.
4. Run inference via
    ```shell
    decode.py \
        --feats-scp /path/to/feats.scp \
        --prompt-scp /path/to/prompt.scp \
        --num-frames /path/to/utt2num_frames \
        --checkpoint /path/to/checkpoint \
        --outdir /path/to/output/wav \
        --verbose ${verbose}  # logging level. higher is more logging. (default=1)
    ```

## Training

First, you need to properly construct `data` and `feats` directory. Please check out [data_prep](data_prep.md) for details.
> 💡Note: here we provide the 16kHz version of model and data. Meanwhile, the original paper uses 24kHz data, which was accomplished by using the features extracted in 16kHz and increasing the `upsample_scales` in the config yaml.


Then, training on LibriTTS (all training partitions) can be done by
```shell
bash run.sh --stage 2 --stop_stage 2 
# You can provide different config file by --conf $your_config
# Checkout run.sh for all the parameters. You can specify every bash variable there as "--key value" in CLI. 
```
This will create `exp/train_all_ctxv2w.v1` for logging.

## Pre-trained model parameters
We release two versions of model parameters (generator) on LibriTTS train-all set. These refer to two sampling rates of the target waveforms. 
Note that the acoustic features (fbank+ppe) are all extracted from 16k waveform. The only difference is the upsample rate in the HifiGAN generator.
* 16k: [link](https://huggingface.co/cantabile-kwok/ctx_vec2wav_libritts_all/resolve/main/ctx_v2w.pkl?download=true)
* 24k: [link](https://huggingface.co/cantabile-kwok/ctx_vec2wav_libritts_all/resolve/main/ctx_v2w_24k.pkl?download=true) with the corresponding config file `conf/ctxv2w.24k.yaml`.

The usage is the same as the "Inference" section. Feel free to use these checkpoints for vocoding!

**CMVN file**: in order to perform inference on out-of-set utterances, we provide the `cmvn.ark` file [here](https://huggingface.co/cantabile-kwok/ctx_vec2wav_libritts_all/resolve/main/cmvn.ark). You should extract mel-spectrogram, normalize by that file (computed on LibriTTS), and then feed the model.

## Acknowledgement
During the development, the following repositories were referred to:
* [ESPnet](https://github.com/espnet/espnet), for most network modules in `ctx_vec2wav/models/conformer`.
* [Kaldi](https://github.com/kaldi-asr/kaldi), for most utility scripts in `utils/`.
* [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), whose training and decoding pipeline is adopted.  

## Citation
```
@article{du2023unicats,
  title={UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding},
  author={Du, Chenpeng and Guo, Yiwei and Shen, Feiyu and Liu, Zhijun and Liang, Zheng and Chen, Xie and Wang, Shuai and Zhang, Hui and Yu, Kai},
  journal={arXiv preprint arXiv:2306.07547},
  year={2023}
}
```

