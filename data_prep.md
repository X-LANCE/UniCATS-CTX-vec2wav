# Data Preparation
In this repo, we rely on the Kaldi-style data formatting. 
We take the LibriTTS (including `clean` and `other` partitions) for example.

### `data/` directory: data manifests
We have organized the `data` directory containing all the LibriTTS data. Here are the steps to establish the `data` dir.
1. Please download from [here](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/data.zip) (about 5MB; or [here](https://www.modelscope.cn/api/v1/datasets/CantabileKwok/libritts-all-kaldi-data/repo?Revision=master&FilePath=data.zip) for Mainland Chinese users), and unzip it to `data` in the project root. Every sub-directory contains `utt2spk, spk2utt` and `wav.scp` files. They are all plain texts, with `<key> <value>` in each line.
2. As we are using the 16kHz version of LibriTTS, please down-sample the speech data if you don't have them. 
3. Then, change the paths in `wav.scp` to the correct ones in your machine.

### `feats/` directory: speech features
We include three types of speech features in CTX-vec2wav. They should all be stored in `feats/` directory in project root.
* **VQ index (together with codebook) from vq-wav2vec**. We extracted it by [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#vq-wav2vec), 
and we provide the extracted VQ index sequences with codebook online.
  1. Please download from [here](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/vqidx.zip) (460MB; [here](https://www.modelscope.cn/api/v1/datasets/CantabileKwok/libritts-all-kaldi-data/repo?Revision=master&FilePath=vqidx.zip) for Chinese users).
  2. Unzip it to `feats/vqidx`, and change the corresponding paths in the `feats.scp`. 
  3. You can check out the feature shape by `feat-to-shape.py scp:feats/vqidx/eval_all/feats.scp | head`. The shapes should be `(frames, 2)`.
* **PPE auxiliary features**. PPE stands for probability of voice, pitch and energy (all in log scale). We extracted them using Kaldi and, to avoid you from installing Kaldi, we provide the extracted and normalized features online.
  1. Please download from [here](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/normed_ppe.zip) (570MB; [here](https://www.modelscope.cn/api/v1/datasets/CantabileKwok/libritts-all-kaldi-data/repo?Revision=master&FilePath=normed_ppe.zip) for Chinese users).
  2. Similarly, please unzip it to `feats/normed_ppe`, and change the corresponding paths in `feats.scp`. 
  3. Check: the shapes of these features should be `(frames, 3)`.
* **Mel spectrograms (FBanks)**. As they are too large, we provide a script to extract them locally:
```shell
nj=64  # parallel jobs. Set this according to your CPU cores.
bash extract_fbank.sh --nj $nj --stage 0 --stop_stage 1  # Default: 80-dim with 10ms frame shift
# Stage 0 extracts fbank in parallel. Stage 1 performs normalization.
```
This will create `feats/fbank` and `feats/normed_fbank` each about 16GB. You can delete `feats/fbank` after normalization.

After having the three types of features, run the following to concatenate these features to form the 85-dim input to the model:
```shell
nj=64
bash dump_feats.sh --nj $nj
```
This will create `feats/dump` and write `feats.scp` in each of the `data` sub-directories. See `exp/dump_feats` for logs. 
As this will create a copy of all the features above, you may delete the rest if you need (**but keep the `feats/vqidx/codebook.npy` safe!**).

Finally, you have correctly formatted the data for training!