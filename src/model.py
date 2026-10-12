import torch
import torch.nn as nn
from transformers import AutoModel


class DINOClassifier(nn.Module):
    """
    Classificatore canonico basato su DINOv3 (ViT-S/16) con Linear Probing.
    Supporta:
      - Congelamento del backbone
      - Estrazione di hidden_states (per Linear-CAM, Eigen-CAM, ecc.)
      - Estrazione di attention weights (per DINO Attention Map)
      - attn_implementation flessibile (es. 'eager')
    """
    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        num_classes=57,
        freeze_backbone=False,
        dropout=0.3,
        attn_implementation=None
    ):
        super().__init__()
        kwargs = {}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        self.backbone = AutoModel.from_pretrained(model_name, **kwargs)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, pixel_values, return_hidden_state=False, return_attentions=False):
        outputs = self.backbone(pixel_values=pixel_values, output_attentions=return_attentions)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)

        if return_hidden_state and return_attentions:
            return logits, outputs.last_hidden_state, outputs.attentions
        if return_hidden_state:
            return logits, outputs.last_hidden_state
        if return_attentions:
            return logits, outputs.attentions
        return logits


class DINOObjectDetector(nn.Module):
    def __init__(self, num_classes, num_queries=100, use_v3=False, unfreeze_last_n=0):
        super().__init__()

        # --- SELEZIONE MODELLO ---
        if use_v3:
            # DINOv3 (Gated / In attesa di approvazione su HF)
            model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        else:
            # DINOv2 Small (Pubblico e subito disponibile su HF)
            # Ha hidden_size = 384, identico a dinov3-vits16!
            model_name = "facebook/dinov2-with-registers-small"

        print(f"--> Caricamento Backbone: {model_name}")

        # Salviamo il VERO modello neurale come backbone
        self.backbone = AutoModel.from_pretrained(model_name)

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            # DINOv3: self.backbone.model.layer | DINOv2: self.backbone.encoder.layer
            if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layer"):
                layers = self.backbone.model.layer  # DINOv3ViTModel
            elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                layers = self.backbone.encoder.layer  # Dinov2Model (with-registers)
            else:
                raise AttributeError("Non trovo la ModuleList dei layer, stampa self.backbone per controllare")

            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

            # LayerNorm finale: "norm" su DINOv3, "layernorm" su DINOv2
            final_norm = getattr(self.backbone, "norm", None) or getattr(self.backbone, "layernorm", None)
            if final_norm is not None:
                for param in final_norm.parameters():
                    param.requires_grad = True

        # 3. Estrazione della dimensione di embedding dalla configurazione (384)
        embed_dim = self.backbone.config.hidden_size

        # 4. Head di Detection (Transformer Decoder stile DETR)
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Output Heads
        self.class_head = nn.Linear(embed_dim, num_classes + 1)  # +1 per background
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid()
        )

        self.multiscale_layers = [3, 6, 9,12]  # indici in hidden_states;
        embed_dim = self.backbone.config.hidden_size
        self.scale_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in self.multiscale_layers])
        self.scale_proj = nn.Linear(embed_dim * len(self.multiscale_layers), embed_dim)

    def forward(self, pixel_values):
        # pixel_values ha forma: [Batch_Size, 3, H, W]

        # 1. Estraiamo le feature da DINO (senza calcolare i gradienti del backbone)
        outputs = self.backbone(pixel_values, output_hidden_states=True)
        num_registers = getattr(self.backbone.config, "num_register_tokens", 0)

        selected = [
            norm(outputs.hidden_states[i][:, (1 + num_registers):, :])
            for norm, i in zip(self.scale_norms, self.multiscale_layers)
        ]
        patch_features = self.scale_proj(torch.cat(selected, dim=-1))

        # 2. Prepariamo le query per il Decoder
        batch_size = pixel_values.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # 3. Passiamo tutto nel Transformer Decoder
        hs = self.decoder(tgt=queries, memory=patch_features)  # Forma: [B, Num_Queries, 384]

        # 4. Predizioni finali
        pred_logits = self.class_head(hs)
        pred_boxes = self.bbox_head(hs)

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
