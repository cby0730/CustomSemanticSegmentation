# loss.py
import segmentation_models_pytorch as smp

def get_loss(loss_name, **kwargs):
    """
    根據 loss_name 建立對應的 loss function 實例。
    
    參數：
      loss_name (str): Loss function 的名稱，可接受的選項包括：
                       "jaccard", "dice", "tversky", "focal", "lovasz", 
                       "softbce", "softcrossentropy", "mcc"
      kwargs: 傳入 loss function 的額外參數。
      
    回傳：
      loss function 實例。
    """
    loss_name = loss_name.lower()
    
    if loss_name == "jaccard":
        # JaccardLoss 支援 mode, classes, from_logits 等參數
        return smp.losses.JaccardLoss(**kwargs)
    
    elif loss_name == "dice":
        return smp.losses.DiceLoss(**kwargs)
    
    elif loss_name == "tversky":
        return smp.losses.TverskyLoss(**kwargs)
    
    elif loss_name == "focal":
        # FocalLoss 不接受 from_logits 與 classes
        kwargs.pop("from_logits", None)
        kwargs.pop("classes", None)
        return smp.losses.FocalLoss(**kwargs)
    
    elif loss_name == "lovasz":
        # LovaszLoss 不接受 classes
        kwargs.pop("classes", None)
        return smp.losses.LovaszLoss(**kwargs)
    
    elif loss_name == "softbce":
        # SoftBCEWithLogitsLoss 不接受 mode, classes, from_logits
        kwargs.pop("mode", None)
        kwargs.pop("classes", None)
        kwargs.pop("from_logits", None)
        return smp.losses.SoftBCEWithLogitsLoss(**kwargs)
    
    elif loss_name == "softcrossentropy":
        # SoftCrossEntropyLoss 不接受 mode, classes, from_logits
        kwargs.pop("mode", None)
        kwargs.pop("classes", None)
        kwargs.pop("from_logits", None)
        # 若 smooth_factor 未指定，預設為 0.0 避免除法錯誤
        if kwargs.get("smooth_factor", None) is None:
            kwargs["smooth_factor"] = 0.0
        return smp.losses.SoftCrossEntropyLoss(**kwargs)
    
    elif loss_name == "mcc":
        # MCCLoss 僅接受 eps，不接受 mode, classes, from_logits
        kwargs.pop("mode", None)
        kwargs.pop("classes", None)
        kwargs.pop("from_logits", None)
        return smp.losses.MCCLoss(**kwargs)
    
    else:
        raise ValueError(f"Loss '{loss_name}' is not supported. Supported losses are: "
                         "jaccard, dice, tversky, focal, lovasz, softbce, softcrossentropy, mcc.")

# 也直接匯出各個 loss 類別，方便其他模組直接引用
JaccardLoss = smp.losses.JaccardLoss
DiceLoss = smp.losses.DiceLoss
TverskyLoss = smp.losses.TverskyLoss
FocalLoss = smp.losses.FocalLoss
LovaszLoss = smp.losses.LovaszLoss
SoftBCEWithLogitsLoss = smp.losses.SoftBCEWithLogitsLoss
SoftCrossEntropyLoss = smp.losses.SoftCrossEntropyLoss
MCCLoss = smp.losses.MCCLoss
