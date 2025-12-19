import torch


def make_optimizer_1stage(cfg, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = cfg.SOLVER.STAGE1.BASE_LR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
    if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE1.MOMENTUM)
    elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)
    return optimizer



def make_optimizer_2stage(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue   
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center


def make_optimizer_2stage_frezee(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "dat" not in key and "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if "dat" not in key and 'image_encoder' in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        print(key)
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center


def make_optimizer_2stage_dat_and_prompt(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "dat" not in key and "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if "dat" not in key and 'image_encoder' in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        print(key)
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center


def make_optimizer_train_prompt_only(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "cv_embed" not in key:
            value.requires_grad_(False)
            continue
        print(key)
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center


def make_optimizer_stage3(cfg, model, center_criterion):
    """
    Create optimizer for Stage 3 training with DFGS sampler.
    
    Stage 3 uses a 10x smaller learning rate than Stage 2 for fine-tuning
    with hard samples mined by the DFGS sampler.
    
    Same parameter freezing strategy as Stage 2:
    - Freeze text_encoder (except DAT modules)
    - Freeze prompt_learner
    - Freeze image_encoder (except DAT modules)
    """
    params = []
    keys = []
    for key, value in model.named_parameters():
        # Freeze text encoder except DAT modules
        if "dat" not in key and "text_encoder" in key:
            value.requires_grad_(False)
            continue
        # Freeze prompt learner
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        # Freeze image encoder except DAT modules
        if "dat" not in key and 'image_encoder' in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        
        # Use Stage 3 learning rate (10x smaller than Stage 2)
        lr = cfg.SOLVER.STAGE3.BASE_LR
        weight_decay = cfg.SOLVER.STAGE3.WEIGHT_DECAY
        
        # Adjust learning rate for bias terms
        if "bias" in key:
            lr = cfg.SOLVER.STAGE3.BASE_LR * getattr(cfg.SOLVER.STAGE3, 'BIAS_LR_FACTOR', 2)
            weight_decay = cfg.SOLVER.STAGE3.WEIGHT_DECAY_BIAS
        
        # Optionally use larger learning rate for classifier layers
        large_fc_lr = getattr(cfg.SOLVER.STAGE3, 'LARGE_FC_LR', False)
        if large_fc_lr:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.STAGE3.BASE_LR * 2
                print('Stage 3: Using two times learning rate for fc')
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    
    print(f"Stage 3 optimizer: {len(params)} parameter groups")
    print(f"Stage 3 base LR: {cfg.SOLVER.STAGE3.BASE_LR}")
    
    optimizer_name = getattr(cfg.SOLVER.STAGE3, 'OPTIMIZER_NAME', 'Adam')
    if optimizer_name == 'SGD':
        momentum = getattr(cfg.SOLVER.STAGE3, 'MOMENTUM', 0.9)
        optimizer = torch.optim.SGD(params, momentum=momentum)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(
            params, 
            lr=cfg.SOLVER.STAGE3.BASE_LR, 
            weight_decay=cfg.SOLVER.STAGE3.WEIGHT_DECAY
        )
    else:
        optimizer = getattr(torch.optim, optimizer_name)(params)
    
    # Center loss optimizer
    center_lr = getattr(cfg.SOLVER.STAGE3, 'CENTER_LR', cfg.SOLVER.STAGE2.CENTER_LR)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=center_lr)
    
    return optimizer, optimizer_center