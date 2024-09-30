import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from argparse import ArgumentParser
from nanogcg import GCGConfig
import rewGCG.nanogcg

args = ArgumentParser()

args.add_argument("data", type=str)
args.add_argument("--question", default="question", type=str)
args.add_argument("--target", default="target", type=str)
args.add_argument("--contrast", type=str)
args.add_argument("--device", default="cuda", type=str)
args.add_argument("--rew-model", default='Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback', type=str)
args.add_argument("--cache-dir", default= '../nobackup/models', type=str)
args.add_argument("--out-dir", type=str)

args = args.parse_args()


model = AutoModelForSequenceClassification.from_pretrained(
    args.rew_model,
    device_map=args.device,
    torch_dtype=torch.bfloat16,
    num_labels=1,
    cache_dir=args.cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(args.rew_model, cache_dir=args.cache_dir)

with open(args.data) as f:
    data = json.load(f)

suffixes = {}
for conversation in data:
    prompt_adv = conversation[args.question] + "{optim_str}"
    target = conversation[args.target]

    messages_adv = [
    {"role": "user", "content": prompt_adv},
    {"role": "assistant", "content": target}
    ]

    if args.contrast:
        contr = conversation[args.contrast]
        message_contrast = [  
        {"role": "user", "content": prompt_adv},
        {"role": "assistant", "content": contr}    
        ]
    else:
        message_contrast = None

    config = GCGConfig(
    num_steps = 250,
    use_prefix_cache=False,
    batch_size=1024,
    allow_non_ascii=False,
    topk = 1024
    )

    res = rewGCG.nanogcg.run(model, tokenizer, messages_adv, contrastive_messages=message_contrast, config=config)
    suffixes[conversation[args.question]] = res.best_string

if args.out_dir:
    with open(args.out_dir, "w") as f:
        json.dump(suffixes, f)



