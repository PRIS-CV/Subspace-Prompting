import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import yaml

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES

from .promptsrc import PromptSRC
from .promptsrc import CustomCLIP as CustomCLIP_
from .promptsrc import VLPromptLearner as VLPromptLearner_
from .supr import load_clip_to_cpu
from .supr import TextEncoder
_tokenizer = _Tokenizer()




class VLPromptLearner(VLPromptLearner_):
    def __init__(self, cfg, classnames, clip_model, templates):
        super().__init__(cfg, classnames, clip_model)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        #Prepare space prompts for SuPr
        space_dim = cfg.TRAINER.SUPR.SPACE_DIM
        if cfg.TRAINER.PROMPTSRC.CTX_INIT:
            with open(cfg.TRAINER.SUPR.HARD_PROMPT_PATH + 'init.yaml', 'r') as file:
              space_init = yaml.load(file, Loader=yaml.FullLoader)
            self.ctx_space = nn.ParameterList([])
            for i in range(space_dim):
                ctx_init = space_init[i]
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                self.ctx_space.append(nn.Parameter(ctx_vectors))
        else:
            # random initialization   
            self.ctx_space = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype)) 
                                               for _ in range(space_dim)])
            for single_para in self.ctx_space:
                nn.init.normal_(single_para, std=0.02) 

        # Prepare the hard prompt embeddings
        hard_prompt_feature = []
        
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        for temp in templates:
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            prompts = prompts.to(torch.device("cuda"))

            with torch.no_grad():
                text_features = clip_model_temp.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            hard_prompt_feature.append(text_features.clone().detach())
        
        self.space_dim = space_dim
        self.hard_prompt_feature = torch.stack(hard_prompt_feature)
        
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, space_dim, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, space_dim, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, space_dim, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]


        all_ctx = [ctx]
        for i in range(self.space_dim):
            all_ctx.append(self.ctx_space[i].unsqueeze(0).expand(self.n_cls, -1, -1))
        ctx = torch.stack(all_ctx, dim=1)


        prompts = torch.cat(
            [
                prefix,  # (n_cls, space_dim+1, 1, dim)
                ctx,     # (n_cls, space_dim+1, n_ctx, dim)
                suffix,  # (n_cls, space_dim+1, *, dim)
            ],
            dim=2,
        )

        return prompts

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix.unsqueeze(1).expand(-1, self.space_dim + 1, -1, -1)
        suffix = self.token_suffix.unsqueeze(1).expand(-1, self.space_dim + 1, -1, -1)

        prompts = self.construct_prompts(ctx, prefix, suffix)
 
        return prompts 

    

class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model, templates):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, templates)
        self.text_encoder = TextEncoder(clip_model)

        self.space_dim = cfg.TRAINER.SUPR.SPACE_DIM
        self.ce_weight = cfg.TRAINER.SUPR.LAMBDA
        self.use_svd = cfg.TRAINER.SUPR.SVD    


    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
  
        # Compute the prompted image and text features
        prompts = self.prompt_learner() #(n_cls * space_dim+1, n_ctx, dim)
        text_features = self.text_encoder(prompts, tokenized_prompts)#(n_cls, n_ctx, dim)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)        
        
        text_features_space = text_features[:,1:,:]
        text_features = text_features[:,0,:]
        # Compute the prompted logits (PromptSRC)
        logits = logit_scale * image_features @ text_features.t()

        #SuPr part
        projected_image_feature = self.project_space(image_features.unsqueeze(1).expand(-1, self.prompt_learner.n_cls, -1),\
                                                     text_features_space) # n_query n_classes n_dim    
        cos_sim = torch.nn.CosineSimilarity(dim=2,eps=1e-07)
        # Compute the space logits (SuPr)
        logits_space = logit_scale * cos_sim(image_features.unsqueeze(1).float(),projected_image_feature)       
        

        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
            
            #SuPr part
            hard_prompt_feature = self.prompt_learner.hard_prompt_feature # T C D
            projected_hardtext_feature = self.project_space(hard_prompt_feature, text_features_space)
                                       
            
            return F.cross_entropy(logits,label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits, logits_space, \
                   F.cross_entropy(logits_space, label), \
                   F.cosine_embedding_loss(hard_prompt_feature.flatten(0,1), projected_hardtext_feature.flatten(0,1), 
                                                torch.ones(hard_prompt_feature.flatten(0,1).size(0)).to(label.device), margin=0.0), \
                  
                   
        else:
            return self.ce_weight * logits_space + (1 - self.ce_weight) * logits
        
    def project_space(self, z_query, z_support):
        # Work on support vectors
        # the shape of z_support is [n_classes, n_support, n_dim]
        #come half, trans float() for inverse,
        z_support = z_support.float()
        z_query = z_query.float()

        # use svd or not to calculate the projection
        if self.use_svd:
   
            z_support = z_support.permute(0,2,1) #n_classes n_dim n_support

            try:# avoid dependency between support vectors
                u, s, v = torch.linalg.svd(z_support, full_matrices=False)
            except:
                u, s, v = torch.linalg.svd(z_support +  1e-4 * torch.randn_like(z_support),full_matrices=False)
            z_support = u
            # Work on query vectors
            # N_0 maybe the number of images or the number of hard prompts embedding
            # z_query [N_0 n_classes n_dim]
            # n_classes,  n_support, n_dim * n_classes, n_dim, N_0 = n_classes, n_support, N_0
            self.beta_hat = torch.matmul(z_support.transpose(1,2), z_query.permute(1,2,0))  
            z_lrc = torch.matmul(z_support,self.beta_hat) 
            return z_lrc.permute(2,0,1)
        
        else: #use least square to calculate the projection
            try:# avoid dependency between support vectors
                z_supports_inv = torch.matmul(torch.linalg.inv(
                    torch.matmul(z_support, z_support.transpose(1, 2))), z_support)# n_classes, n_support, n_dim
            except:
                z_supports_inv = torch.matmul(torch.linalg.inv(
                    torch.matmul(z_support, z_support.transpose(1, 2)) +  1e-4 * torch.eye(  #n_classes, n_support, n_support
                        z_support.shape[1],).cuda().repeat(z_support.shape[0], 1, 1)), z_support)# n_classes, n_support, n_dim
            
            beta_hat = torch.matmul(z_supports_inv, z_query.permute(1, 2, 0))  #  [n_classes, n_support, n_dim] * [n_classes, n_dim, N_0] = [n_classes, n_support, N_0]
            z_lrc = torch.matmul(z_support.transpose(1, 2), beta_hat)  # [n_classes, n_dim, n_support] * [n_classes, n_support, N_0] = n_classes, n_dim, T

            return z_lrc.permute(2,0,1)
    
 

@TRAINER_REGISTRY.register()
class SubspacePromptSRC(PromptSRC):
    """
    Subspace Prompting for PromptSRC
    """
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        all_classnames = [name.replace("_", " ") for name in self.dm.dataset.all_classnames]
        max_name_len = max([len(_tokenizer.encode(name)) for name in all_classnames]) + 2  #'. EOS'
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg,False, max_name_len=max_name_len)
       
        if cfg.TRAINER.PROMPTSRC.PREC == "fp32" or cfg.TRAINER.PROMPTSRC.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        with open(cfg.TRAINER.SUPR.HARD_PROMPT_PATH + 'genertic_templates.yaml', 'r') as file:
              genertic_hard_prompt = yaml.load(file, Loader=yaml.FullLoader)
        templates = genertic_hard_prompt #
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, templates) 

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        # Keep model with GPA
        self.previous_model_gpa = None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits, logits_space, loss_ce_space, loss_hard_reg = model(image, label)
            # Calculate the L_SCL_text loss
            loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                      reduction='mean')
            # Calculate the L_SCL_image loss
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                       reduction='mean')
                                       
            # Now calculate L_SCL_logits
            L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            L_SCL_logits_space = F.kl_div(
                F.log_softmax(logits_space / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            L_SCL = 0.7 * L_SCL_logits_space + 0.15 * L_SCL_logits + \
                    loss_scl_image * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT + \
                    loss_scl_text  * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT 
            loss = self.cfg.TRAINER.SUPR.LAMBDA * loss_ce_space + \
                    (1 - self.cfg.TRAINER.SUPR.LAMBDA) * loss_ce + \
                    + loss_hard_reg * self.cfg.TRAINER.SUPR.REG_LOSS_WEIGHT + L_SCL
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
            "loss_ce_point": loss_ce.item(),
            "loss_ce_space": loss_ce_space.item(),
            "loss_hard_reg": loss_hard_reg.item(),
            "loss_SCL": L_SCL.item()
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

        if self.step_counter == self.model.total_epochs + 1:
            print("Using GPA model for final inference...")
            state_dict = self.previous_model_gpa
            filtered_state_dict = state_dict
            model.load_state_dict(filtered_state_dict, strict=False)
            self.model.load_state_dict(filtered_state_dict, strict=False)
            
        return loss_summary
