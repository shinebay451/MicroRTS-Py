import argparse
import time

from classes.DataCollector import DecisionTransformerGymDataCollator
from classes.TrainableDT import TrainableDT
from datasets import DatasetDict
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for DT GridNet")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to stored dataset')
    parser.add_argument('--num-train-epochs', type=int,
                        default=120, help='Number of training epochs')
    parser.add_argument('--per-device-train-batch-size', type=int,
                        default=16, help='Batch size per device during training')
    parser.add_argument('--learning-rate', type=float,
                        default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=1e-4, help='Weight decay')
    parser.add_argument('--warmup-ratio', type=float,
                        default=0.1, help='Warmup ratio')
    parser.add_argument('--optim', type=str,
                        default="adamw_torch", help='Optimizer to use')
    parser.add_argument('--max-grad-norm', type=float,
                        default=0.25, help='Maximum gradient norm for clipping')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Use mixed precision training')
    parser.add_argument('--logging-steps', type=int,
                        default=10, help='how often to log to wandb')
    parser.add_argument('--scheduler', type=str,
                        default="linear", help='Learning rate scheduler')
    parser.add_argument('--model-path', type=str,
                        help='Path to model to load')
    parser.add_argument('--save-dir', type=str,
                        help='Directory to save the model')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        dataset = DatasetDict.load_from_disk(args.dataset)
    except:
        raise FileNotFoundError("Dataset not found")
    collector = DecisionTransformerGymDataCollator(dataset["train"])

    if args.model_path:
        model = TrainableDT.from_pretrained(args.model_path)
        model.train()
    else:
        config = DecisionTransformerConfig(
            state_dim=collector.state_dim,
            act_dim=collector.act_dim,
            hidden_size=1024,
            n_head=16,
            n_layer=12,
        )
        model = TrainableDT(config)

    agent_name = args.dataset.split("/")[-2]
    if args.model_path:
        path = args.model_path
        if path[-1] == "/":
            path = path[:-1]
        output_dir = "/".join(path.split("/")[0:-1])
    elif args.save_dir:
        output_dir = f"{args.save_dir}/dt-{agent_name}-{time.time()}".replace(".", "")
    else:
        output_dir = f"models/dt-{agent_name}-{time.time()}".replace(".", "")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_steps=args.logging_steps,
        save_steps=50,
        remove_unused_columns=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.mixed_precision,
        lr_scheduler_type=args.scheduler,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collector,
    )

    if args.model_path:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
