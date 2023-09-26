# Implementation of CTX-vec2wav 

> This is the official implementation of CTX-vec2wav vocoder in the paper [UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding](https://arxiv.org/abs/2306.07547)

<img width="1187" alt="image-20230926140022539" src="https://github.com/cantabile-kwok/CTX-vec2wav/assets/58417810/036708e0-90a0-4df6-a886-3c1b3ba47e29">


### Acknowledgement
During the development, the following repositories are referred:
* [sooftware/conformer](https://github.com/sooftware/conformer), whose conformer implementation is directly used in `ctx_vec2wav/models/conformer`.
* [ESPnet](https://github.com/espnet/espnet), for some network modules in `ctx_vec2wav/models/conformer`.
* [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), whose training and decoding pipeline is adopted.  

### Citation
```
@article{du2023unicats,
  title={UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding},
  author={Du, Chenpeng and Guo, Yiwei and Shen, Feiyu and Liu, Zhijun and Liang, Zheng and Chen, Xie and Wang, Shuai and Zhang, Hui and Yu, Kai},
  journal={arXiv preprint arXiv:2306.07547},
  year={2023}
}
```

