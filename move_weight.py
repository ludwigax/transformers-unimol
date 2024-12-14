import torch

cache = torch.load("pretrained/dptech_unimol/mol_pre_all_h_220816.pt")['model']

new_cache = {}
for key, value in cache.items():
    if "layers" in key:
        key = key.replace("encoder", "unimol")
        if "in_proj" in key:
            q, k, v = value.chunk(3, dim=0)
            q_proj = key.replace("in_proj", "q_proj")
            k_proj = key.replace("in_proj", "k_proj")
            v_proj = key.replace("in_proj", "v_proj")
            
            new_cache[q_proj] = q
            new_cache[k_proj] = k
            new_cache[v_proj] = v
        elif "out_proj" in key:
            o_proj = key.replace("out_proj", "o_proj")

            new_cache[o_proj] = value
        elif "fc1" in key:
            fc1 = key.replace("fc1", "mlp.up_proj")
            new_cache[fc1] = value
        elif "fc2" in key:
            fc2 = key.replace("fc2", "mlp.down_proj")
            new_cache[fc2] = value
        elif "self_attn_layer_norm" in key:
            layer_norm = key.replace("self_attn_layer_norm", "input_layernorm")
            new_cache[layer_norm] = value
        elif "final_layer_norm" in key:
            layer_norm = key.replace("final_layer_norm", "post_attention_layernorm")
            new_cache[layer_norm] = value

    elif "embed_tokens" in key:
        key = key.replace("embed_tokens", "unimol.embed_tokens")
        new_cache[key] = value[0: 30, ...]

    elif "emb_layer_norm" in key:
        key = key.replace("encoder.emb_layer_norm", "unimol.embed_layernorm")
        new_cache[key] = value

    elif "final_layer_norm" in key:
        key = key.replace("encoder.final_layer_norm", "unimol.post_layernorm")
        new_cache[key] = value

    elif "gbf" in key:
        if "linear1" in key:
            key = key.replace("gbf_proj.linear1", "unimol.pair_attn.proj.up_proj")
            new_cache[key] = value
        elif "linear2" in key:
            key = key.replace("gbf_proj.linear2", "unimol.pair_attn.proj.down_proj")
            new_cache[key] = value
        elif "means" in key:
            means = "unimol.pair_attn.gbf.means"
            new_cache[means] = value.squeeze(0)
        elif "stds" in key:
            stds = "unimol.pair_attn.gbf.stds"
            new_cache[stds] = value.squeeze(0)

        elif "mul" in key:
            mul = "unimol.pair_attn.gbf.mul.weight"
            new_cache[mul] = value[0:900, ...]

        elif "bias" in key:
            bias = "unimol.pair_attn.gbf.bias.weight"
            new_cache[bias] = value[0:900, ...]

for key, value in new_cache.items():
    print(f"{key}:\t{value.shape}\t{value.dtype}")

torch.save(new_cache, "moved_weight.pt")