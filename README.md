# \[Working in Progress\] CTX-vec2wav, the Acoustic Context-Aware Vocoder

> This is the official implementation of CTX-vec2wav vocoder in the paper [UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding](https://arxiv.org/abs/2306.07547).

<img width="1187" alt="image-20230926140022539" src="https://github.com/cantabile-kwok/CTX-vec2wav/assets/58417810/036708e0-90a0-4df6-a886-3c1b3ba47e29">

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
Working in Progress

## Training

First, you need to properly construct `data` and `feats` directory. Please check out [data_prep](data_prep.md) for details.

Then, training on LibriTTS (all training partitions) can be done by
```shell
bash run.sh --stage 2 --stop_stage 2 
# You can provide different config file by --conf $your_config
# Checkout run.sh for all the parameters. You can specify every bash variable there as "--key value" in CLI. 
```
This will create `exp/train_all_ctxv2w.v1` for logging.

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

