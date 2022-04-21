from multilingual.models.xpt_neo.modeling_xpt_neo import XPTNeoForCausalLM


def print_model(_model):
    for name, module in _model.named_modules():
        if hasattr(module, "attention_type"):
            print(
                name,
                module.__class__.__qualname__,
                module.attention_type,
                module.attention.q_proj.weight.requires_grad,
            )


model = XPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

model.initialize_xpt(
    bos_token_id=0,
    eos_token_id=0,
    new_embedding_size=40000,
    num_itl_layers=2,
)

print_model(model)
