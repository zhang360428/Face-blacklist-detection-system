# 构建人脸库黑名单并实现基于黑名单的人脸识别拦截

使用流程如下：

## conda环境

使用environment.yml文件生成FaceSafety环境，conda activate激活该环境

## milvus数据库

使用milvus-docker目录下的docker-compose文件拉取docker镜像，已经针对国内网络问题替换了docker源
注意该版本使用的是gpu版本的milvus，如果需要cpu版本，需要去https://github.com/milvus-io/milvus/releases 官网获取对应的docker-compose文件

## 数据

数据我们采用的是kaggle上的开源数据集来实验。
需要去https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download 下载并解压替换archive文件夹

build_blacklist.py用于对该数据集建立黑名单人脸库（只出现一次的人脸直接进测试集，出现一个人多张人脸图片的情况，则将第一张图片进入黑名单人脸库其他图片则作为测试集使用）
evaluate.py用于验证测试集上的效果，对于正负样本分别的拦截情况
