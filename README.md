# Mujoco Playground实践

实践Mujoco Playground的安装、dm_control_suite和locomotion的测试。

![](https://gitclone.com/download1/playground/sample0.gif)
![](https://gitclone.com/download1/playground/sample1.gif)

## 一、环境要求

GPU：至少24G显存，因为训练时需要18G左右显存。如RTX4090、RTX3090等

操作系统：Linux，如Ubuntu22.04

## 二、环境安装

### 1、安装ffmpeg

```shell
sudo apt update
sudo apt install -y ffmpeg
```

### 2、创建虚拟环境

```shell
conda create -n mujoco python==3.12 -y
conda activate mujoco
```

### 3、安装依赖库

```shell
# clone源码
git clone https://github.com/git-cloner/mujoco_playground_sample
# 切换目录
cd mujoco_playground_sample
# 安装依赖库
pip install -r requirements.txt \
-i https://mirrors.aliyun.com/pypi/simple
```

### 4、安装JAX GPU版

```shell
pip install "jax[cuda12]" -f \
https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# 验证
python -c "import jax; print(jax.default_backend())" 
```

## 三、运行测试程序

```shell
# 测试dm_control_suite
CUDA_VISIBLE_DEVICES=0 python dm_control_suite.py
# 测试运动
CUDA_VISIBLE_DEVICES=0 python locomotion.py
```

## 四、参考

```shell
# 官网例子
git clone https://github.com/google-deepmind/mujoco_playground
cd mujoco_playground/learning/notebooks
# jupyter 转 python
jupyter nbconvert --to python dm_control_suite.ipynb
# 其他参考文件
https://research.mels.ai/ide?mels=UnitreeGo1.qkazy
```




