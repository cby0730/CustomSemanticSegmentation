# optimizer.py
import torch.optim as optim

def get_optimizer(optimizer_name, parameters, lr, **kwargs):
    """
    根據 optimizer_name 建立對應的優化器實例。
    
    參數：
      optimizer_name (str): 優化器名稱，可選項包括：
                            "adam", "adamw", "sgd", "radam", "rmsprop", "adadelta", "adagrad"
      parameters: 要優化的模型參數（通常為 model.parameters()）。
      lr (float): 學習率。
      kwargs: 傳入優化器的其他參數，例如 weight_decay、momentum 等。
      
    回傳：
      優化器實例。
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr, **kwargs)
    elif optimizer_name == "adamw":
        return optim.AdamW(parameters, lr=lr, **kwargs)
    elif optimizer_name == "sgd":
        return optim.SGD(parameters, lr=lr, **kwargs)
    elif optimizer_name == "radam":
        # torch.optim.RAdam 需確認 PyTorch 版本（v1.13 以上支援）
        return optim.RAdam(parameters, lr=lr, **kwargs)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, **kwargs)
    elif optimizer_name == "adadelta":
        return optim.Adadelta(parameters, lr=lr, **kwargs)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(parameters, lr=lr, **kwargs)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Supported optimizers are: adam, adamw, sgd, radam, rmsprop, adadelta, adagrad.")

# 也可直接匯出各個優化器類別，方便其他模組直接引用
Adam = optim.Adam
AdamW = optim.AdamW
SGD = optim.SGD
RAdam = getattr(optim, "RAdam", None)  # 若環境支援 RAdam 則不為 None
RMSprop = optim.RMSprop
Adadelta = optim.Adadelta
Adagrad = optim.Adagrad
