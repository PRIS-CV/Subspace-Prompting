import collections
import os.path as osp
import os


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from torch import linalg as LA
import random
from tqdm import tqdm
import yaml
import copy

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import ResidualAttentionBlock_SuPr

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False, max_name_len=6):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": cfg.TRAINER.SUPR.TRAINER_BACKBONE,
                          "vision_depth": cfg.TRAINER.SUPR.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.SUPR.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.SUPR.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.SUPR.N_CTX_TEXT,
                          "space_dim": cfg.TRAINER.SUPR.SPACE_DIM,
                          "max_name_len": max_name_len}
    else:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, 
                          "vision_ctx": 0,
                          "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts):
        #(n_cls, space_dim+1, n_ctx, dim)
        x = prompts + self.positional_embedding.type(self.dtype)
        n_cls, s, n_ctx, dim, = x.size()
       
        x = self.transformer(x)
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eot = tokenized_prompts.argmax(dim=-1).view(n_cls, 1, 1, 1).expand(n_cls, s, 1, dim).to(x.device)

        x = torch.gather(x, dim=2, index=eot) @ self.text_projection
       
        return x.squeeze(2)
    
       

class SubspacePromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, templates):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.SUPR.N_CTX_TEXT
        space_dim = cfg.TRAINER.SUPR.SPACE_DIM
        ctx_init = cfg.TRAINER.SUPR.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        assert cfg.TRAINER.SUPR.PROMPT_DEPTH_TEXT >= 0, "For SuPr, PROMPT_DEPTH should be >= 1, 1 is shallow prompting"
        self.text_prompts_depth = cfg.TRAINER.SUPR.PROMPT_DEPTH_TEXT  # max=12, but will create 11 such shared prompts
        self.vision_prompts_depth = cfg.TRAINER.SUPR.PROMPT_DEPTH_VISION  # max=12, but will create 11 such shared prompts

     
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            self.ctx = nn.Parameter(ctx_vectors)
            prompt_prefix = ctx_init

            with open(cfg.TRAINER.SUPR.HARD_PROMPT_PATH + 'init.yaml', 'r') as file:
                space_init = yaml.load(file, Loader=yaml.FullLoader)
            self.ctx_space = nn.ParameterList([]) #self.ctx_space
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
            self.ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype))
            nn.init.normal_(self.ctx, std=0.02)       
            prompt_prefix = " ".join(["X"] * n_ctx)
            self.ctx_space = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype)) 
                                               for _ in range(space_dim)]) #ctx_space
            for single_para in self.ctx_space:
                nn.init.normal_(single_para, std=0.02) 

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
       
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]


        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

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
       
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.space_dim = space_dim
        self.tokenized_prompts = tokenized_prompts  
        
        self.name_lens = name_lens
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
            all_ctx.append(self.ctx_space[i].unsqueeze(0).expand(self.n_cls, -1, -1))#ctx_space
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


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, templates):
        super().__init__()
        
        self.prompt_learner = SubspacePromptLearner(cfg, classnames, clip_model, templates)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype     

        self.space_dim = cfg.TRAINER.SUPR.SPACE_DIM
        self.use_svd = cfg.TRAINER.SUPR.SVD
        self.ce_weight = cfg.TRAINER.SUPR.LAMBDA    # balance coeficient for two logits, gamma in the paper   

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts    
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner() #(n_cls * space_dim+1, n_ctx, dim)
        text_features = self.text_encoder(prompts, tokenized_prompts)#(n_cls, n_ctx, dim)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)        
        
        text_feature_point = text_features[:,0,:]
        text_features = text_features[:,1:,:]
        projected_image_feature = self.project_space(image_features.unsqueeze(1).expand(-1, self.prompt_learner.n_cls, -1),text_features) # n_query n_classes n_dim
    
        cos_sim = torch.nn.CosineSimilarity(dim=2,eps=1e-07)
        logits = logit_scale * cos_sim(image_features.unsqueeze(1).float(),projected_image_feature)       
        logits_point = logit_scale * image_features @ text_feature_point.t()

        if self.prompt_learner.training:
            hard_prompt_feature = self.prompt_learner.hard_prompt_feature # template, n_cls, dim
            projected_hardtext_feature = self.project_space(hard_prompt_feature, text_features)

            return logits, F.cross_entropy(logits, label), \
                    F.cross_entropy(logits_point, label), \
                    F.cosine_embedding_loss(hard_prompt_feature.flatten(0,1), projected_hardtext_feature.flatten(0,1), 
                                                torch.ones(hard_prompt_feature.flatten(0,1).size(0)).to(label.device), margin=0.0) 
                   
        else:
            return self.ce_weight * logits + (1 - self.ce_weight) * logits_point
   
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
    
 
class CustomCLIP_Ens(nn.Module):
    def __init__(self, cfg, classnames, templates, all_classnames, ensemble_num):
        super().__init__()
        self.ensemble_num = ensemble_num
        # distribute templates to each model
        split_templates = [templates[i::ensemble_num] for i in range(ensemble_num)]
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")  

        #calculate the max length of class names for concatenation
        all_classnames = [name.replace("_", " ") for name in all_classnames]
        max_name_len = max([len(_tokenizer.encode(name)) for name in all_classnames]) + 2  #'. EOS'
       
        ens_clip_model = [load_clip_to_cpu(cfg,False, max_name_len=max_name_len) for _ in range(ensemble_num)] #replace new VPT for ensemble model
        if cfg.TRAINER.SUPR.PREC == "fp32" or cfg.TRAINER.SUPR.PREC == "amp":
            # CLIP's default precision is fp16
            ens_clip_model = [clip_model.float() for clip_model in ens_clip_model]

        # share the frozen parameters for all models
        for i in range(1,ensemble_num):
            for name, param in ens_clip_model[i].named_parameters():          
                if "VPT" not in name:   
                    module = ens_clip_model[i]
                    module_shared = ens_clip_model[0]
                    modules = name.split('.')
                    if len(modules)>1:
                        for module_name in modules[:-1]:
                            module = getattr(module, module_name)         
                            module_shared = getattr(module_shared, module_name)
                    module_shared = getattr(module_shared, modules[-1])
                    setattr(module, modules[-1], module_shared) 
                                
        self.ensemble_model = nn.ModuleList([CustomCLIP(cfg, classnames, ens_clip_model[i], split_templates[i]) 
                                             for i in range(ensemble_num)])
                                            
    def forward(self, image, label=None):
        results = [model(image, label) if label is not None else model(image)
                   for model in self.ensemble_model]  
        if label is not None:
            stacked_results = [
                torch.stack([r[i] for r in results]).mean(0)
                for i in range(len(results[0]))  
            ]
            return tuple(stacked_results)
        return torch.stack(results).mean(0)

@TRAINER_REGISTRY.register()
class SuPrEns(TrainerX):
    """Supspace Prompting with Ensemble
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.SUPR.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print("Building custom CLIP")
        with open(cfg.TRAINER.SUPR.HARD_PROMPT_PATH + 'genertic_templates.yaml', 'r') as file:
              genertic_hard_prompt = yaml.load(file, Loader=yaml.FullLoader)
        templates = genertic_hard_prompt #+ specific_hard_pormpt

        assert cfg.TRAINER.SUPR.ENSEMBLE_NUM>1, f"Ensemble number should >1, 1 for SuPr, else for SuPr-Ens"
        self.model = CustomCLIP_Ens(cfg, classnames, templates,
                                         self.dm.dataset.all_classnames,cfg.TRAINER.SUPR.ENSEMBLE_NUM)


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


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)


        ####
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # However, IMAGENET need multiple GPU for 1000 class text prompt (3090TI)
        self.device = torch.device("cuda:0")
        self.device1 = torch.device("cuda")
        self.model.to(self.device)
        for ensemble_model in self.model.ensemble_model:
            ensemble_model.text_encoder=nn.DataParallel(ensemble_model.text_encoder.to(self.device1))
       
        

        # NOTE: only give prompt_learner to the optimizer
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("SubspacePromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.SUPR.PREC == "amp" else None

   
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.SUPR.PREC
        if prec == "amp":
            with autocast():
                output, loss_ce, loss_ce_point, loss_hard_reg = model(image, label)
                loss = self.cfg.TRAINER.SUPR.LAMBDA * loss_ce + (1 - self.cfg.TRAINER.SUPR.LAMBDA) * loss_ce_point \
                    + loss_hard_reg * self.cfg.TRAINER.SUPR.REG_LOSS_WEIGHT
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            output, loss_ce, loss_ce_point, loss_hard_reg = model(image, label)
            loss = self.cfg.TRAINER.SUPR.LAMBDA * loss_ce + (1 - self.cfg.TRAINER.SUPR.LAMBDA) * loss_ce_point \
                + loss_hard_reg * self.cfg.TRAINER.SUPR.REG_LOSS_WEIGHT
            self.model_backward_and_update(loss)
     

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "loss_ce": loss_ce.item(),
            "loss_ce_point": loss_ce_point.item(),
            "loss_hard_reg": loss_hard_reg.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        
        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

    
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            for s in state_dict.keys():
                if "token_prefix" in s:
                    print(s)
            
            # Ignore fixed token vectors
            for i in range(50):
                if "ensemble_model."+str(i)+".prompt_learner.token_prefix" in state_dict:
                    del state_dict["ensemble_model."+str(i)+".prompt_learner.token_prefix"]
                if "ensemble_model."+str(i)+".prompt_learner.token_suffix" in state_dict:
                    del state_dict["ensemble_model."+str(i)+".prompt_learner.token_suffix"]


            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


