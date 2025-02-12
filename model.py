# model.py
"""
model.py

本模組提供一個函式 get_model()，可根據所選模型名稱與參數，
動態建立 smp 的預訓練 segmentation 模型。
支援的模型包括：
    - Unet
    - Unet++ (也接受 "unet++" 或 "unetplusplus")
    - FPN
    - PSPNet
    - DeepLabV3
    - DeepLabV3+ (也接受 "deeplabv3+" 或 "deeplabv3plus")
    - Linknet
    - MAnet
    - PAN
    - UPerNet
    - Segformer

使用範例：
    from model import get_model
    model = get_model(
                model_name="unet",
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=5,
                activation=None
            )
"""

import segmentation_models_pytorch as smp

def get_model(model_name, encoder_name="resnet34", encoder_weights="imagenet",
              in_channels=3, classes=1, activation=None, **kwargs):
    """
    根據模型名稱建立一個 segmentation model。

    Parameters:
        model_name (str): 模型名稱，支援的選項：
                          "unet", "unet++", "fpn", "pspnet", "deeplabv3",
                          "deeplabv3+", "linknet", "manet", "pan", "upernet", "segformer"
        encoder_name (str): 選用的 encoder/backbone 名稱（預設 "resnet34"）。
        encoder_weights (str or None): Encoder 預訓練權重（例如 "imagenet" 或 None）。
        in_channels (int): 輸入通道數（預設 3）。
        classes (int): 分類數（對應 mask 的類別數）。
        activation (str or callable or None): 最後一層後的 activation（預設 None，通常與 CrossEntropyLoss 配合）。
        **kwargs: 傳入模型建構函式的其他參數（例如 encoder_depth、decoder_segmentation_channels 等）。

    Returns:
        torch.nn.Module: 指定的 segmentation 模型實例。
    """
    model_name = model_name.lower()
    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name in ["unet++", "unetplusplus"]:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name == "fpn":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name == "pspnet":
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name in ["deeplabv3+", "deeplabv3plus"]:
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name == "linknet":
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name == "manet":
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name == "pan":
        model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name == "upernet":
        model = smp.UPerNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    elif model_name == "segformer":
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Supported models are: "
                         "unet, unet++, fpn, pspnet, deeplabv3, deeplabv3+, linknet, manet, pan, upernet, segformer.")
    return model

if __name__ == "__main__":
    # 測試範例：建立一個 Unet 模型並輸出模型結構
    model = get_model(
        model_name="unet",
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
        activation=None
    )
    print(model)
