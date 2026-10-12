import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ==========================================
# 1. UTILITY GEOMETRICHE (Box & GIoU)
# ==========================================

def box_cxcywh_to_xyxy(x):
    """ Converte box da [cx, cy, w, h] a [x1, y1, x2, y2] """
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """ Calcola IoU tra due insiemi di box """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """ Calcola GIoU (Generalized IoU) per misurare la distanza tra box anche se non sovrapposte """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# ==========================================
# 2. HUNGARIAN MATCHER (Algoritmo Ungherese)
# ==========================================

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: dict("pred_logits": [B, N, num_classes + 1], "pred_boxes": [B, N, 4])
        targets: list di B dizionari, ognuno con {"labels": [M], "boxes": [M, 4]}
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten delle predizioni per calcolare la matrice dei costi
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*N, num_classes + 1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*N, 4]

        # Concatena tutti i target del batch
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if len(tgt_ids) == 0:
            # Caso limite: nessuna box reale nel batch
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) for _ in range(bs)]

        # 1. Costo di Classificazione: -prob[target_class]
        cost_class = -out_prob[:, tgt_ids]

        # 2. Costo L1 Bounding Box
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 3. Costo GIoU
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Matrice dei Costi Totale
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            if sizes[i] == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
            else:
                # Algoritmo Ungherese eseguito su CPU via SciPy
                src, tgt = linear_sum_assignment(c[i])
                indices.append((torch.as_tensor(src, dtype=torch.int64), torch.as_tensor(tgt, dtype=torch.int64)))

        return indices


# ==========================================
# 3. SET CRITERION (La Loss Totale DETR)
# ==========================================

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        """
        num_classes: numero di POI (es. 57)
        matcher: istanza di HungarianMatcher
        weight_dict: pesi delle loss {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
        eos_coef: peso per la classe "no object / background" (default 0.1 per evitare sbilanciamento)
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        # Pesi delle classi per la CrossEntropy (riduce la penalizzazione sul background)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef  # L'ultima classe è "no object"
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        # Mappa (batch_idx, query_idx) per estrarre le predizioni accoppiate
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        """ Loss di classificazione (Cross-Entropy con no-object) """
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        # Crea la matrice di target per tutte le N=100 query, inizializzata a "no object" (indice num_classes)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )

        # Assegna le classi reali solo alle query accoppiate dall'Hungarian Matcher
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """ Loss sulle Bounding Box (L1 + GIoU) solo per le query accoppiate """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]

        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(target_boxes) == 0:
            return {"loss_bbox": torch.tensor(0.0, device=src_boxes.device),
                    "loss_giou": torch.tensor(0.0, device=src_boxes.device)}

        # 1. L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes

        # 2. GIoU Loss
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou.sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        # 1. Calcola l'accoppiamento ottimale tra Query e Target
        indices = self.matcher(outputs, targets)

        # Conteggio box totali per normalizzazione
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # 2. Calcola le singole componenti di Loss
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        # 3. Applica i pesi e somma
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)
        return total_loss, losses