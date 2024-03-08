import re
import shutil
import argparse

from Deepseek.configuration_deepseek import DeepseekConfig
from Deepseek.modeling_deepseek import DeepseekForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the input model")
parser.add_argument("--output_path", type=str, help="Path to save the output model")
parser.add_argument("--n_routed_experts", type=int, help="Number of routed experts")
parser.add_argument("--num_experts_per_tok", type=int, help="Number of experts per token")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

dense_config = AutoConfig.from_pretrained(args.model_path).to_dict()
dense_config["n_routed_experts"] = args.n_routed_experts
dense_config["num_experts_per_tok"] = args.num_experts_per_tok
dense_config["moe_intermediate_size"] = dense_config["intermediate_size"]
dense_config["norm_topk_prob"] = True
dense_config["auto_map"] = {
    "AutoConfig": "configuration_deepseek.DeepseekConfig",
    "AutoModel": "modeling_deepseek.DeepseekModel",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekForCausalLM"
}
dense_model = AutoModelForCausalLM.from_pretrained(args.model_path)

sparse_config = DeepseekConfig(**dict(dense_config))
sparse_model = DeepseekForCausalLM(sparse_config)

param_dict = dict(dense_model.named_parameters())
for name, param in sparse_model.named_parameters():
    if name in param_dict:
        param.data.copy_(param_dict[name].data)
    elif re.sub(r'experts\.\d+\.', '', name) in param_dict:
        param.data.copy_(param_dict[re.sub(r'experts\.\d+\.', '', name)].data)
    else:
        continue

tokenizer.save_pretrained(args.output_path)
sparse_model.save_pretrained(args.output_path)
shutil.copy("Deepseek/configuration_deepseek.py", args.output_path)
shutil.copy("Deepseek/modeling_deepseek.py", args.output_path)
