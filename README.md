# About Diffusers

For the math of transformer models, materials can be found easily in many places. Here, the article dysmystified some important topics in transformer models, https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/. Another article presented in-depth descriptions on the mathematics adopted by Stable Diffusion models, e.g., SD 1.x and SD 2.x, and this article can be found in https://lilianweng.github.io/posts/2021-07-11-diffusion-models/, and its authors also presented other topics as well.

Some articles were listed here for reference, their content was primaryly about the arithmetic issues related to transformer models. 

As for the arithmetic intensiry required to carry out transformer inference, some details have been presented in an article, https://kipp.ly/transformer-inference-arithmetic/. Besides the discussion on FLOPs, extra analysis was made on the memory costs as well. In addition, reference can be found in another artile, https://www.adamcasson.com/posts/transformer-flops.

In analysis, the roof-line model has always been used, and in https://www.baseten.co/blog/llm-transformer-inference-guide/, there is some analysis work on LLM inference and performance. Furthermore, benchmarking tests were conducted to figure out the inference performance of LLM models, as shown in https://www.baseten.co/blog/benchmarking-fast-mistral-7b-inference/. 


It might be expected to train the stable diffusion models, and descriptions can be found in https://github.com/explainingai-code/StableDiffusion-PyTorch.

The code snippet in test_sdxl1-base.py shows how to load the SD.v1-5 model (from https://blog.csdn.net/qq_38423499/article/details/137158458).

The article (from https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16) presented a conversion tool, examples, and instructions on how to set up Stable Diffusion with ONNX model. Nvidia also provided their own ways to achieve the same purpose of transverting a PyToch script into a model in ONNX format, and details can be found in https://developer.nvidia.com/zh-cn/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/. 


About the compilation developed for PyTorch 2, please take a loot at the article presented in https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26. There its authors presented a brief overview of the compilation procedure, especially the IR related descriptions. 

A in-depth explanation had been presented and it was titled with "How Pytorch 2.0 Accelerates Deep Learning with Operator Fusion and CPU/GPU Code-Generation", which can be found in https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26. 

An article stated the concept of attention and its role in text generation and LLM, and its content can be found in https://lilianweng.github.io/posts/2018-06-24-attention/.

Here in https://onceuponanalgorithm.org/guide-what-is-a-stable-diffusion-seed-and-how-to-use-it/, desriptions have been provided for "seed", which is seldom discussed in other places.
