"""
Optimizer builder for EVA-CLIP ReID model
Handles both Stage 1 (prompt learning) and Stage 2 (adapter training)
"""

import torch


def make_optimizer_1stage(cfg, model):
    """
    Stage 1 optimizer: Only train prompt learner
    Purpose: Align text prompts with visual features using contrastive loss
    """
    params = []
    keys = []
    
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = cfg.SOLVER.STAGE1.BASE_LR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
            print(f"[Stage1] Training: {key}")
    
    print(f"[Stage1] Total trainable parameters: {len(keys)}")
    
    if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.STAGE1.MOMENTUM)
    elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, 
                                       weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)
    
    return optimizer


def make_optimizer_2stage(cfg, model, center_criterion):
    """
    Stage 2 optimizer: Train adapters, classifiers, and other components
    Freeze: text_encoder, prompt_learner, EVA-CLIP backbone
    Train: adapters (CFAA, IFA), classifiers, bottleneck, cv_embed
    """
    params = []
    keys = []
    
    # Keywords to identify trainable components
    adapter_keywords = ['cfaa', 'ifa', 'adapter', 'dat']
    trainable_keywords = ['classifier', 'bottleneck', 'cv_embed']
    freeze_keywords = ['text_encoder', 'prompt_learner']
    
    for key, value in model.named_parameters():
        # Freeze text encoder
        if any(kw in key.lower() for kw in freeze_keywords):
            value.requires_grad_(False)
            continue
        
        # Check if this is part of image encoder (EVA-CLIP backbone)
        if 'image_encoder' in key or 'eva_vit' in key or 'visual' in key:
            # Only train adapter components within image encoder
            if not any(kw in key.lower() for kw in adapter_keywords):
                value.requires_grad_(False)
                continue
        
        if not value.requires_grad:
            continue
        
        # Set learning rate
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.STAGE2.BASE_LR * 2
                print(f'Using 2x learning rate for: {key}')
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
        print(f"[Stage2] Training: {key}")
    
    print(f"[Stage2] Total trainable parameters: {len(keys)}")
    
    # Create optimizer
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, 
                                       weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    
    # Center loss optimizer
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), 
                                        lr=cfg.SOLVER.STAGE2.CENTER_LR)
    
    return optimizer, optimizer_center


def make_optimizer_2stage_adapters_and_prompt(cfg, model, center_criterion):
    """
    Stage 2 optimizer variant: Train adapters AND prompts
    Useful when you want to continue fine-tuning prompts in stage 2
    """
    params = []
    keys = []
    
    adapter_keywords = ['cfaa', 'ifa', 'adapter', 'dat']
    trainable_keywords = ['classifier', 'bottleneck', 'cv_embed', 'prompt_learner']
    
    for key, value in model.named_parameters():
        # Freeze text encoder (but not prompt learner)
        if 'text_encoder' in key:
            value.requires_grad_(False)
            continue
        
        # Check if this is part of image encoder
        if 'image_encoder' in key or 'eva_vit' in key or 'visual' in key:
            if not any(kw in key.lower() for kw in adapter_keywords):
                value.requires_grad_(False)
                continue
        
        if not value.requires_grad:
            continue
        
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        
        # Different LR for different components
        if "prompt_learner" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * 0.1  # Lower LR for prompts
        elif "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        elif cfg.SOLVER.STAGE2.LARGE_FC_LR and ("classifier" in key or "arcface" in key):
            lr = cfg.SOLVER.STAGE2.BASE_LR * 2
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
        print(f"[Stage2+Prompt] Training: {key}")
    
    print(f"[Stage2+Prompt] Total trainable parameters: {len(keys)}")
    
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, 
                                       weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), 
                                        lr=cfg.SOLVER.STAGE2.CENTER_LR)
    
    return optimizer, optimizer_center


def make_optimizer_2stage_freeze_backbone(cfg, model, center_criterion):
    """
    Stage 2 optimizer: Freeze entire EVA-CLIP backbone, only train adapters and heads
    Most memory efficient option
    """
    params = []
    keys = []
    
    adapter_keywords = ['cfaa', 'ifa', 'adapter', 'dat']
    head_keywords = ['classifier', 'bottleneck', 'cv_embed']
    
    for key, value in model.named_parameters():
        # Freeze text encoder and prompt learner
        if 'text_encoder' in key or 'prompt_learner' in key:
            value.requires_grad_(False)
            continue
        
        # Freeze entire image encoder except adapters
        if 'image_encoder' in key or 'eva_vit' in key or 'visual' in key:
            if not any(kw in key.lower() for kw in adapter_keywords):
                value.requires_grad_(False)
                continue
        
        # Only keep adapters and heads trainable
        is_trainable = (
            any(kw in key.lower() for kw in adapter_keywords) or
            any(kw in key.lower() for kw in head_keywords)
        )
        
        if not is_trainable:
            value.requires_grad_(False)
            continue
        
        if not value.requires_grad:
            continue
        
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
        print(f"[Stage2-Frozen] Training: {key}")
    
    print(f"[Stage2-Frozen] Total trainable parameters: {len(keys)}")
    
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, 
                                       weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), 
                                        lr=cfg.SOLVER.STAGE2.CENTER_LR)
    
    return optimizer, optimizer_center


def make_optimizer_cv_embed_only(cfg, model, center_criterion):
    """
    Train only camera/view embeddings (PBP)
    Useful for quickly adapting to new camera setups
    """
    params = []
    keys = []
    
    for key, value in model.named_parameters():
        if "cv_embed" not in key:
            value.requires_grad_(False)
            continue
        
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
        print(f"[CV-Embed] Training: {key}")
    
    print(f"[CV-Embed] Total trainable parameters: {len(keys)}")
    
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, 
                                       weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), 
                                        lr=cfg.SOLVER.STAGE2.CENTER_LR)
    
    return optimizer, optimizer_center


def get_trainable_params_info(model):
    """
    Utility function to print information about trainable parameters
    """
    total_params = 0
    trainable_params = 0
    
    print("\n" + "="*80)
    print("PARAMETER SUMMARY")
    print("="*80)
    
    # Group by component
    components = {}
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Determine component
        if 'image_encoder' in name or 'eva_vit' in name or 'visual' in name:
            if 'cfaa' in name.lower() or 'ifa' in name.lower() or 'adapter' in name.lower():
                component = 'adapters'
            else:
                component = 'backbone'
        elif 'text_encoder' in name:
            component = 'text_encoder'
        elif 'prompt_learner' in name:
            component = 'prompt_learner'
        elif 'classifier' in name:
            component = 'classifier'
        elif 'bottleneck' in name:
            component = 'bottleneck'
        elif 'cv_embed' in name:
            component = 'cv_embed'
        else:
            component = 'other'
        
        if component not in components:
            components[component] = {'total': 0, 'trainable': 0}
        
        components[component]['total'] += param.numel()
        if param.requires_grad:
            components[component]['trainable'] += param.numel()
            trainable_params += param.numel()
    
    for comp_name, info in components.items():
        status = "✓" if info['trainable'] > 0 else "✗"
        print(f"{status} {comp_name:20s}: {info['trainable']:>12,} / {info['total']:>12,} trainable")
    
    print("-"*80)
    print(f"{'Total':20s}: {trainable_params:>12,} / {total_params:>12,} trainable")
    print(f"Trainable percentage: {100.0 * trainable_params / total_params:.2f}%")
    print("="*80 + "\n")
    
    return total_params, trainable_params