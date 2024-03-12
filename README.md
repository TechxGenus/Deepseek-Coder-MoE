# Deepseek-Coder-MoE
Sparse Deepseek-Coder.

Perform sparse upcycling on dense models:
```sh
python sparse_upcycling.py \
--model_path deepseek-ai/deepseek-coder-6.7b-base \
--output_path merged \
--n_routed_experts 8 
--num_experts_per_tok 2
```

Convert the Mixtral architecture model to the DeepseekMoE architecture (for models obtained using the model fusion method):

```sh
python convert_from_mixtral.py \
--model_path deepseek-ai/deepseek-coder-6.7b-base \
--output_path merged
```
