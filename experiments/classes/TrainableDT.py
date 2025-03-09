import torch
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

        loss = torch.mean((actions_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)
