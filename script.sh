#!/bin/bash
# run_experiments.sh
# 此腳本用以遍歷不同的模型、encoder、loss 與 optimizer 組合進行實驗，
# 並根據 optimizer 特性自動設定相對應的 learning rate。

# 模型選項
models=("segformer") # "unet" "unet++" "fpn" "pspnet" "deeplabv3" "deeplabv3+" "linknet" "manet" "pan" "upernet" 

# 非 segformer 模型允許的 encoder（排除 mit_b0, mit_b1, mit_b2）
non_mit_encoders=("resnet18" "resnet34" "resnet50" "resnext50_32x4d" \
"vgg11" "vgg11_bn" "vgg13" "vgg13_bn" "vgg16" "vgg16_bn" "vgg19" "vgg19_bn" \
"densenet121" "densenet169" "densenet201" "densenet161" \
"mobilenet_v2" "mobileone_s0" "mobileone_s1" "mobileone_s2" "mobileone_s3" "mobileone_s4" \
"efficientnet-b0" "efficientnet-b1" "efficientnet-b2" "efficientnet-b3" "efficientnet-b4" "efficientnet-b5" \
"timm-efficientnet-b0" "timm-efficientnet-b1" "timm-efficientnet-b2" "timm-efficientnet-b3" "timm-efficientnet-b4" "timm-efficientnet-b5" \
"timm-tf_efficientnet_lite0" "timm-tf_efficientnet_lite1" "timm-tf_efficientnet_lite2" "timm-tf_efficientnet_lite3" "timm-tf_efficientnet_lite4")

# 若模型為 segformer，只允許使用 mit 系列
mit_encoders=("mit_b0" "mit_b1" "mit_b2")

# Loss 函式選項（多類別分割使用）
losses=("crossentropy" "jaccard" "dice" "tversky" "focal" "lovasz" "softcrossentropy")

# Optimizer 選項
optimizers=("adam" "adamw" "sgd" "radam" "rmsprop" "adadelta" "adagrad")

# 其他固定參數（可依需求修改）
train_data_dir="data/train"
valid_data_dir="data/valid"
epochs=100
batch_size=32
num_workers=4
save_interval=50
encoder_weights="imagenet"

# 主迴圈：遍歷所有組合
for model in "${models[@]}"; do
    # 根據模型選擇 encoder 清單：segformer 只搭 mit 系列，其它禁止使用 mit 系列 encoder
    if [ "$model" == "segformer" ]; then
        encoders=("${mit_encoders[@]}")
    else
        encoders=("${non_mit_encoders[@]}")
    fi

    for encoder in "${encoders[@]}"; do
        for loss_fn in "${losses[@]}"; do
            for optimizer in "${optimizers[@]}"; do
                # 根據 optimizer 的特性自動設定 learning rate
                if [ "$optimizer" == "sgd" ]; then
                    lr=0.01
                elif [ "$optimizer" == "adadelta" ]; then
                    lr=1.0
                elif [ "$optimizer" == "adagrad" ]; then
                    lr=0.01
                else
                    lr=1e-4
                fi

                # 為每個組合建立獨立的 checkpoint 資料夾，方便區分不同實驗
                ckpt_dir="checkpoints/${model}_${encoder}_${loss_fn}_${optimizer}"
                mkdir -p "$ckpt_dir"
                echo "=========================="
                echo "Running experiment:"
                echo "  Model:      $model"
                echo "  Encoder:    $encoder"
                echo "  Loss:       $loss_fn"
                echo "  Optimizer:  $optimizer"
                echo "  Learning Rate: $lr"
                echo "  Checkpoints dir: $ckpt_dir"
                echo "--------------------------"
                python train.py \
                  --train_data_dir "$train_data_dir" \
                  --valid_data_dir "$valid_data_dir" \
                  --checkpoints_dir "$ckpt_dir" \
                  --epochs "$epochs" \
                  --batch_size "$batch_size" \
                  --num_workers "$num_workers" \
                  --learning_rate "$lr" \
                  --save_interval "$save_interval" \
                  --model_name "$model" \
                  --encoder_name "$encoder" \
                  --encoder_weights "$encoder_weights" \
                  --loss_fn "$loss_fn" \
                  --optimizer "$optimizer"
            done
        done
    done
done
