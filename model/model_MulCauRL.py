import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tokenizer import build_tokenizer
from utils.GeneSetEncoder import GeneEncoder
from cra_modules.m_cra.only_trans import LanModel
from cra_modules.m_cra.modules4vlci1 import VDM, LDM, FDIntervention,GDM


# ------------------------------
#   Attention subnets (clam1)
# ------------------------------
class Attn_Net(nn.Module):
    """Attention Network without Gating (2 fc layers)"""

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        module = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            module.append(nn.Dropout(0.25))
        module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        # returns (attn_logits, passthrough_x)
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """Attention Network with Sigmoid Gating (3 fc layers)"""

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        a = [nn.Linear(L, D), nn.Tanh()]
        b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            a.append(nn.Dropout(0.25))
            b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*a)
        self.attention_b = nn.Sequential(*b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)  # N x n_classes
        return A, x


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from models.attention import Attn_Net, Attn_Net_Gated
# from models.language import build_tokenizer, LanModel
# from models.causal import VDM, LDM, FDIntervention
# from models.genes import GeneEncoder


class MulCauRL(nn.Module):

    def __init__(
        self,
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=10,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        subtyping=False,
        embed_dim=1024,
        args=None,
        vd_topk=6,
        vd_bins=49,
        use_ldm=True,
        dim_text=32,
        dim_gene=32,  # fixed final gene dim = 32
    ):
        super().__init__()

        # ---- dims ----
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        L_in, _, D_att = self.size_dict[size_arg]
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        # ---- attention backbone ----
        fc = [nn.Linear(L_in, 512), nn.ReLU(), nn.Dropout(dropout)]
        attn_cls = Attn_Net_Gated if gate else Attn_Net
        fc.append(attn_cls(L=512, D=D_att, dropout=dropout, n_classes=n_classes))
        self.attention_net = nn.Sequential(*fc)

        # ===== multimodal / causal =====
        self.dim_D = 512
        self.dim_T = getattr(args, "dim_text", dim_text) if args is not None else dim_text
        self.dim_G_final = 32
        self.use_ldm = getattr(args, "use_ldm", use_ldm) if args is not None else use_ldm
        self.classifiers_vg = nn.ModuleList([
            nn.Linear(544, 1) for _ in range(self.n_classes)
        ])
        # ===== text side =====
        self._use_text_stack = False
        try:
            self.tokenizer = build_tokenizer(args) if args is not None else None
            if self.tokenizer is not None:
                self.lang_model = LanModel(self.tokenizer,
                                           lan=getattr(args, "lan", "bert"),
                                           out_dim=768)
                self.t_proj = nn.Linear(768, self.dim_T)
                self.t_toD = nn.Linear(self.dim_T, self.dim_D)
                self.ln_t = nn.LayerNorm(self.dim_D)
                self._use_text_stack = True
                for p in self.lang_model.parameters():
                    p.requires_grad = False
                self.lang_model.eval()
        except Exception:
            self.lang_model = None
            self._use_text_stack = False

        # ===== gene side =====
        try:
            self.gene_encoder = GeneEncoder(d_out=self.dim_G_final)
            self.ln_gene = nn.LayerNorm(self.dim_G_final)
        except Exception:
            self.gene_encoder = None
            self.ln_gene = nn.LayerNorm(self.dim_G_final)

        # ===== visual side =====
        self.ln_img = nn.LayerNorm(self.dim_D)

        # ===== causal modules =====
        self.vdm = VDM(embed_dim=self.dim_D, bins=vd_bins, topk=vd_topk)
        self.ldm = LDM(embed_dim=self.dim_D) if self.use_ldm else None
        self.fim_v = FDIntervention(self.dim_D)
        self.fim_w = FDIntervention(self.dim_D)
        self.gdm = GDM(embed_dim_img=512, embed_dim_gene=32)

        # gene mapping 32->512 for front-door (explicit init)
        self.map_gene = nn.Sequential(
            nn.Linear(32, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        self.fim_g = FDIntervention(embed_dim=512)

        # bottleneck 512->32 for fusion
        self.g_proj = nn.Sequential(
            nn.Linear(512, 32, bias=False),
            nn.LayerNorm(32),
        )

        # fused dim = 512 + 512 + 32 = 1056
        fused_dim = self.dim_D * 2 + 32
        self.classifiers = nn.ModuleList([nn.Linear(fused_dim, 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList([nn.Linear(512, 2) for _ in range(n_classes)])

        # ===== regularization weights (can override by args) =====
        self.lambda_car = float(getattr(args, "lambda_car", 1.0))
        self.align_w  = float(getattr(args, "align_w",  0.3))  
        self.causal_w = float(getattr(args, "causal_w", 0.15)) 
        self.ent_w    = float(getattr(args, "ent_w", 0.10))    
        self._eps = 1e-8

        # (optional) vocab projection cache
        self._vocab_toD = nn.Linear(768, self.dim_D, bias=False)
        self._cached_W = None
        self._cached_W_dev = None

    # ===== instance sampling =====
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A_c, h, classifier):
        device = h.device
        if len(A_c.shape) == 1:
            A_c = A_c.view(1, -1)
        top_p_ids = torch.topk(A_c, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A_c, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        loss = self.instance_loss_fn(logits, all_targets)
        return loss, all_preds, all_targets

    def inst_eval_out(self, A_c, h, classifier):
        device = h.device
        if len(A_c.shape) == 1:
            A_c = A_c.view(1, -1)
        top_p_ids = torch.topk(A_c, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        loss = self.instance_loss_fn(logits, p_targets)
        return loss, p_preds, p_targets

    def _get_W_hat(self, device):
        if not self._use_text_stack or self.lang_model is None:
            return torch.zeros(1, self.dim_D, device=device)
        try:
            with torch.no_grad():
                W = self.lang_model.get_vocab_matrix(detach=True)
                if (self._cached_W is None) or (self._cached_W_dev != device):
                    self._cached_W = self._vocab_toD(W.to(device))
                    self._cached_W_dev = device
                return self._cached_W
        except Exception:
            return torch.zeros(1, self.dim_D, device=device)

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
        **kwargs,
    ):
        # ---- attention ----
        A, h_emb = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)

        # ---- bag feature (visual) ----
        M_img = torch.mm(A, h_emb)          # [K, 512]
        M_img = self.ln_img(M_img)

        # ---- local/global contexts ----
        K, N = A.size()
        k_local = min(self.k_sample, N)
        top_idx = torch.topk(A, k=k_local, dim=1).indices
        h_vl = torch.stack([h_emb.index_select(0, top_idx[c]) for c in range(K)], 0)  # [K, k, 512]

        Hg = h_emb.unsqueeze(0)                 # [1, N, 512]
        fg = self.vdm.global_mamba(Hg).expand(K, -1, -1)  # [K, N, 512]
        mt = self.vdm.fuse(h_vl, fg)            # [K, k, 512]
        Mv = F.adaptive_avg_pool1d(mt.transpose(1, 2), 1).transpose(1, 2).squeeze(1)  # [K, 512]

        # ---- text path ----
        description = kwargs.get("description", None)
        #print(description)
        #description=['This is a patient diagnosed with lung cancer.']

        if description is not None and self._use_text_stack:
            with torch.no_grad():
                t768 = self.lang_model(description)
            if t768.dim() == 1:
                t768 = t768.unsqueeze(0)
            t_vec = self.t_proj(t768).squeeze(0)
            h_t = self.ln_t(self.t_toD(t_vec.unsqueeze(0)).squeeze(0))  # [512]
        else:
            h_t = torch.zeros(self.dim_D, device=h_emb.device)
        h_t_exp = h_t.unsqueeze(0).expand(M_img.size(0), -1)  # [K, 512]

        if self.use_ldm and description is not None and self._use_text_stack:
            W_hat = self._get_W_hat(h_emb.device)   # [V, 512]
            #Ml = self.ldm(h_vl, W_hat, h_t_exp)     # [K, 512]
        else:
            W_hat = self._get_W_hat(h_emb.device)
            #Ml = torch.zeros_like(M_img)            # [K, 512]

        # ---- gene path ----
        gene_data = kwargs.get("gene_data", None)
        #print(gene_data)
        #gene_data=['(AKT1,0,1),(ALK,0,1),(ARID1A,0,1),(BRAF,0,1),(CD274,0,1),(CDKN2A,0,1),(CREBBP,0,1),(CXCL13,0,1),(EGFR,0,1),(EP300,0,1),(ERBB2,0,1),(FGFR1,0,1),(HRAS,0,1),(IFNG,0,1),(IRF1,0,1),(JAK2,0,1),(KEAP1,0,1),(KMT2D,0,1),(KRAS,0,1),(MET,0,1),(NF1,0,1),(NFE2L2,0,1),(NLRP3,0,1),(NRAS,0,1),(NTRK1,0,1),(PD-L1,0,1),(PIK3CA,0,1),(PTEN,0,1),(RB1,0,1),(RBM10,0,1),(RET,0,1),(ROS1,0,1),(SETD2,0,1),(SMARCA4,0,1),(SMARCB1,0,1),(SOX2,0,1),(STAT3,0,1),(STK11,0,1),(TP53,0,1)']
        if gene_data is not None and self.gene_encoder is not None:
            g_vec = self.gene_encoder(gene_data)
            if g_vec.dim() == 2 and g_vec.size(0) > 1:
                g_vec = g_vec.mean(0)
            else:
                g_vec = g_vec.squeeze(0)
        else:
            g_vec = torch.zeros(self.dim_G_final, device=h_emb.device)
        g_vec = self.ln_gene(g_vec)                 # [32]
        g_exp = g_vec.unsqueeze(0).expand(M_img.size(0), -1)  # [K, 32]

        M_g = self.gdm(h_vl,h_t_exp, g_exp)            # [K, 512] mediator-guided gene
        z_g_med = self.map_gene(g_exp)             # [K, 512] mapped gene (direct gene->img space)
        z_g = self.fim_g(z_g_med, M_g)             # [K, 512] front-door gene
        z_g32 = self.g_proj(z_g)                   # [K, 32]  bottleneck for fusion

        # ---- front-door visual/text ----
        z_v = self.fim_v(M_img, Mv)                # [K, 512]
        #z_w = self.fim_w(h_t_exp, Ml)              # [K, 512]
        z_v = F.layer_norm(z_v, (z_v.size(-1),))
        #z_w = F.layer_norm(z_w, (z_w.size(-1),))

        # ---- fusion & classifier ----
        M_causal = torch.cat([z_v, h_t_exp, z_g32], dim=-1)      # [K, 1056]
        M_causal = F.layer_norm(M_causal, (M_causal.size(-1),))

        logits = torch.empty(1, self.n_classes, device=M_causal.device, dtype=M_causal.dtype)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M_causal[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        # ---- instance-level eval (optional) ----
        if instance_eval:
            total_inst_loss = 0.0
            all_preds, all_targets = [], []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = int(inst_labels[i].item())
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    loss_i, preds, targets = self.inst_eval(A[i], h_emb, classifier)
                else:
                    if self.subtyping:
                        loss_i, preds, targets = self.inst_eval_out(A[i], h_emb, classifier)
                    else:
                        continue
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
                total_inst_loss += loss_i
            if self.subtyping:
                total_inst_loss /= float(len(self.instance_classifiers))
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}

        # ======== Lightweight regularizations ========
        reg_losses = {}
        
        # (1) Modality alignment (cosine)
        cos_vw = 1 - F.cosine_similarity(z_v, h_t_exp, dim=-1).mean()
        cos_vg = 1 - F.cosine_similarity(z_v, z_g, dim=-1).mean()
        cos_wg = 1 - F.cosine_similarity(h_t_exp, z_g, dim=-1).mean()
        align_loss = (cos_vw + cos_vg + cos_wg) / 3.0
        reg_losses["align_loss"] = align_loss
        
        # =====================================================
        # (2) NEW Causal Consistency Loss (strict front-door)
        # =====================================================
        
        # --- Full front-door versions ---
        z_v_full = z_v         # fim_v(M_img, Mv)
        z_w_full = h_t_exp       # fim_w(h_t_exp, Ml)
        z_g_full = z_g32       # g_proj( fim_g(map_gene, M_g) )
        
        # --- Mediator-removed versions (no front-door) ---
        z_v_med = M_img                     # visual path without mediator
        z_w_med = h_t_exp                   # text path without mediator
        z_g_med = self.g_proj(self.map_gene(g_exp))   # gene without M_g
        
        # --- Fuse for "med" version ---
        M_med = torch.cat([z_v_med, z_w_med, z_g_med], dim=-1)  # [K,1056]
        M_med = F.layer_norm(M_med, (M_med.size(-1),))
        
        # --- Classifier on med version ---
        logits_med = torch.empty(1, self.n_classes, 
                                 device=M_med.device, dtype=M_med.dtype)
        for c in range(self.n_classes):
            logits_med[0, c] = self.classifiers[c](M_med[c])
        
        p_full = F.softmax(logits, dim=1)        # full front-door
        p_med  = F.softmax(logits_med, dim=1)    # mediator removed
        
        causal_consistency = F.kl_div((p_med + self._eps).log(),
                                      p_full,
                                      reduction="batchmean")
        reg_losses["causal_loss"] = causal_consistency
        # (3) Entropy regularization on full prediction
        entropy = -torch.sum(p_full * torch.log(p_full + self._eps))
        reg_losses["entropy_loss"] = entropy
        # sum-up
        total_reg = self.lambda_car * (
              self.align_w  * reg_losses["align_loss"]
            + self.causal_w * reg_losses["causal_loss"]
            + self.ent_w    * reg_losses["entropy_loss"]
        )
        results_dict["aux_reg"] = total_reg
        results_dict.update(reg_losses)


        if return_features:
            results_dict.update({"features": M_causal})

        return logits, Y_prob, Y_hat, A_raw, results_dict



