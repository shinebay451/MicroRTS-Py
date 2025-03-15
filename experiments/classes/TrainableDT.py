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
        return super().forward(**kwargs)
