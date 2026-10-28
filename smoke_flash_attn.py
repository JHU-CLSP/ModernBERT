"""Run from the repo: ../.venv/bin/python smoke_flash_attn.py

Tiny offline MLM forward/backward through the actual repo model; requires CUDA
and FlashAttention 2. Does not launch Composer or load data/checkpoints.
"""

import torch
import flash_attn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from flash_attn.ops.triton.layer_norm import RMSNorm

from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.model import FlexBertForMaskedLM


def main():
    assert torch.cuda.is_available(), "CUDA GPU is not accessible"
    print(f"torch={torch.__version__}, flash_attn={flash_attn.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    torch.manual_seed(0)
    config = FlexBertConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_attention_heads=2, num_hidden_layers=3, max_position_embeddings=257,
        attention_layer="rope", embedding_layer="sans_pos", padding="unpadded",
        unpad_embeddings=True, pad_logits=False, use_fa2=True,
        sliding_window=128, global_attn_every_n_layers=2,
        loss_function="fa_cross_entropy", loss_kwargs={"reduction": "mean"},
        final_norm=True, num_initial_layers=0, compile_model=False,
    )
    model = FlexBertForMaskedLM(config).cuda().train()
    attention = [layer.attn for layer in model.bert.encoder.layers]
    for layer in attention:
        assert layer.use_fa2, "Unexpected SDPA fallback"
        layer.use_fa3 = False  # Test FA2 even if an optional FA3 is installed.
    assert [layer.sliding_window for layer in attention] == [(-1, -1), (64, 64), (-1, -1)]
    assert isinstance(model.loss_fn, CrossEntropyLoss)

    # Unequal lengths, both longer than the local window, with real padding.
    input_ids = torch.randint(1, config.vocab_size, (2, 257), device="cuda")
    mask = torch.arange(257, device="cuda")[None, :] < torch.tensor([257, 129], device="cuda")[:, None]
    input_ids.masked_fill_(~mask, 0)
    labels = input_ids.clone().masked_fill(~mask, -100)
    labels[:, 1::3] = -100  # Exercise ignored MLM labels too.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=input_ids, attention_mask=mask.long(), labels=labels)
    assert output.logits.shape == (386, config.vocab_size), output.logits.shape
    assert torch.isfinite(output.logits).all(), "Nonfinite logits"
    assert torch.isfinite(output.loss), "Nonfinite loss"
    output.loss.backward()
    for layer in attention:
        grad = layer.Wqkv.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, "Invalid QKV gradient"
    torch.cuda.synchronize()
    print(f"PASS: BF16 RoPE + global/local FA2 + fused CE + backward; loss={output.loss.item():.4f}")
    print("Unpadded QKV: (386, 3, 2, 64); local window: (64, 64); logits:", tuple(output.logits.shape))


if __name__ == "__main__":
    main()
