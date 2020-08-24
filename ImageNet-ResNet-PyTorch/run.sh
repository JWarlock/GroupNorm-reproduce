#!/usr/bin/env bash
python main_pytorch_examples.py --arch resnet18 --norm 'GN' --batch-size 128 --lr 0.05 \
--dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 '/data/wangzeyu/ILSRVC/Imagenet2012/'

python main_pytorch_examples.py --arch resnet50 --norm 'GN' --batch-size 128 --lr 0.05 \
--dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 '/data/wangzeyu/ILSRVC/Imagenet2012/'

python main_pytorch_examples.py --arch resnet101 --norm 'GN' --batch-size 128 --lr 0.05 \
--dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 '/data/wangzeyu/ILSRVC/Imagenet2012/'

python main_pytorch_examples.py --arch caffenet -j 16 --batch-size 256 --lr 0.005 \
--dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 '/data/wangzeyu/ILSRVC/Imagenet2012/'

python main_pytorch_examples.py --arch caffenet_bn -j 16 --batch-size 256 --lr 0.05 \
--dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 '/data/wangzeyu/ILSRVC/Imagenet2012/'