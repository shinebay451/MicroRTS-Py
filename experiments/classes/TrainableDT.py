import torch
import torch.nn.functional as F
from transformers import DecisionTransformerModel


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)

        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        actions_preds = action_preds.reshape(-1,
                                             act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1,
                                                act_dim)[attention_mask.reshape(-1) > 0]

        # cross entropy loss
        loss = F.cross_entropy(actions_preds, action_targets)

        return {"loss": loss, "logits": action_preds}

    def original_forward(self, **kwargs):
        output = super().forward(**kwargs)
        action_preds = output[1]
        act_dim = kwargs["actions"].shape[2]
        action_pred = action_preds.reshape(-1, act_dim)[-1]
        action_pred = action_pred.reshape(-1, 78)  # 78 action encodings

        slices = [
            (0, 6),    # action type
            (6, 10),   # move parameter
            (10, 14),  # harvest parameter
            (14, 18),  # return parameter
            (18, 22),  # produce direction parameter
            (22, 29),  # produce type parameter
            (29, 78)   # relative attack position
        ]

        for start, end in slices:
            action_pred[:, start:end] = torch.nn.functional.one_hot(torch.argmax(
                action_pred[:, start:end], dim=1), num_classes=end-start)

        return action_pred.flatten()
