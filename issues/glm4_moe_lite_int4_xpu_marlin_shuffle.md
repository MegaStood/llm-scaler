# GLM-4.7-Flash INT4 MoE: Marlin Shuffle DEVICE_LOST on Lunar Lake XPU

## Status: FIXED — AutoRound FusedMoE routing to IPEX native path

## Problem

When serving `glm-4.7-flash-int4-autoround` via vLLM on Intel Arc 140V (Lunar Lake),
the IPEX Marlin weight shuffle crashes the Level Zero backend during model warmup.

```
RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```

The crash occurs in `_IPEXGatedMLPMOEXPU.__init__()` at:
```
intel_extension_for_pytorch/transformers/models/xpu/fusions/linear_fusion.py:152
  self.W13 = self.marlin_shuffle_weight(self.W13)
```

## Root Cause

The `marlin_shuffle_weight()` function reshuffles INT4 packed weights into Marlin
kernel layout. This happens **lazily at first forward** via `init_on_device()`,
at which point the model (16.52 GB) is already loaded on XPU. The iGPU shares
system memory (30 GB total), leaving very little headroom.

The function iterates over all 65 experts per layer (64 routed + 1 shared),
performing complex bit manipulation on XPU. With 46 MoE layers, this is ~3000
shuffle operations that overwhelm the iGPU.

## Approaches Tested (all failed)

### 1. CPU-side shuffle (whole tensor)
Move entire `qweight` to CPU, shuffle there, move back.
- **Result**: `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` - the `.cpu()` call itself
  needs staging memory that isn't available.

### 2. CPU-side shuffle (per-expert)
Move one expert at a time to CPU via `qweight[e].cpu()`.
- **Result**: `UR_RESULT_ERROR_DEVICE_LOST` - even a single expert transfer
  crashes the device under memory pressure.

### 3. Sync + empty_cache before transfer
Add `torch.xpu.synchronize()` and `torch.xpu.empty_cache()` before CPU transfer.
- **Result**: Still `DEVICE_LOST` - the driver instability isn't from pending ops.

### 4. Pre-shuffle in process_weights_after_loading (on CPU before XPU load)
Perform Marlin shuffle in vLLM's `process_weights_after_loading()` while weights
are still on CPU, then pass pre-shuffled weights to IPEX GatedMLPMOE.
- **Result**: `UR_RESULT_ERROR_OUT_OF_RESOURCES` - the pre-shuffled weights + model
  together exceed available memory when loaded to XPU. The shuffle creates a second
  copy of the weight tensor temporarily.

### 5. IPEX_MOE_GEMM_NATIVE=1 (bypass Marlin, use native GEMM)
Skip Marlin shuffle entirely, use IPEX native MoE GEMM path.
- **Result**: `expected self and mat2 to have the same dtype, but got: c10::BFloat16 != int`
  The native path doesn't dequantize INT4 packed weights - it expects FP16/BF16.

## Fix: AutoRound `apply_ipex_quant_layer` FusedMoE Routing (2026-04-13)

The root cause was that `AutoRoundConfig.apply_ipex_quant_layer()` only handled
`LinearBase`/`ParallelLMHead` layers. When a `FusedMoE` layer was passed:

1. `get_quant_method()` on XPU calls `apply_ipex_quant_layer()` (line 460)
2. `apply_ipex_quant_layer()` checks `isinstance(layer, (LinearBase, ParallelLMHead))` — `FusedMoE` doesn't match
3. Returns `None` — the layer gets no quantization method
4. `FusedMoE` falls back to upstream `GPTQMarlinMoEMethod` (CUDA Marlin path)
5. `GPTQMarlinMoEMethod.process_weights_after_loading()` calls `marlin_shuffle_weight()`
6. `marlin_shuffle_weight()` allocates large temporary tensors on XPU -> **DEVICE_LOST** on shared-memory iGPU

Meanwhile, `IPEXConfig.get_quant_method()` in `ipex_quant.py:225-226` already has
the correct FusedMoE routing to `XPUGPTQMarlinMoEMethod` — it just never got called
for AutoRound models because `apply_ipex_quant_layer` returned `None` first.

### The fix

Added `FusedMoE` handling to `apply_ipex_quant_layer()` in
`vllm/model_executor/layers/quantization/auto_round.py`:

```python
elif isinstance(layer, FusedMoE) and "gptq" in self.packing_format:
    from vllm.model_executor.layers.quantization.ipex_quant import (
        XPUGPTQMarlinMoEMethod,
    )
    config = IPEXConfig(
        method="gptq",
        weight_bits=weight_bits,
        group_size=group_size,
        is_qweight_sym=sym,
        dynamic={},
        full_config={},
    )
    return XPUGPTQMarlinMoEMethod(config, layer.moe_config)
```

This routes all INT4 AutoRound GPTQ MoE models (GLM-4.7-Flash, Gemma 4, etc.)
to the IPEX native `XPUGPTQMarlinMoEMethod` path, which calls
`ipex.llm.modules.GatedMLPMOE()` directly — no Marlin shuffle, no temporary
tensor allocation, no DEVICE_LOST on shared-memory iGPU.

### Affected models
- `Intel/glm-4.7-flash-int4-autoround` (30B-A3B MoE)
- `Intel/gemma-4-26b-a4b-it-int4-autoround` (MoE)
- Any future INT4 AutoRound GPTQ MoE model on XPU

## Previous Investigation (for reference)

The approaches below were tested before the root cause was identified.
They all attempted to work around the Marlin shuffle within the CUDA code path,
which was the wrong path entirely.

## Related Files

| File | Role |
|------|------|
| `intel_extension_for_pytorch/transformers/models/xpu/fusions/linear_fusion.py:151-208` | Marlin shuffle init + function |
| `intel_extension_for_pytorch/xpu/intrinsic/__init__.py:~506` | Native MoE GEMM fallback |
| `vllm/model_executor/layers/quantization/ipex_quant.py:742-757` | vLLM creates IPEX GatedMLPMOE |
| `vllm/model_executor/layers/fused_moe/layer.py:1995` | MoE forward dispatch |

## Environment
- Intel Core Ultra 7 258V (Lunar Lake), Arc 140V iGPU
- 32 GB LPDDR5x shared memory
- vLLM 0.14.1.dev0, IPEX XPU
- Model: glm-4.7-flash-int4-autoround (16.52 GB loaded)
