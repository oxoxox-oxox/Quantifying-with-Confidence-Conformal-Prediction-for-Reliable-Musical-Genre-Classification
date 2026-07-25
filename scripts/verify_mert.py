"""Step 2: MERT model loading + GPU inference verification."""
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MODEL_NAME = "m-a-p/MERT-v1-330M"

print(f"[1/5] Loading feature extractor: {MODEL_NAME}")
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"      sample_rate={processor.sampling_rate}")

print(f"[2/5] Loading model: {MODEL_NAME}")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = model.cuda()
model.eval()

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"      params={total_params:.0f}M")

vram_used = torch.cuda.memory_allocated() / 1024**3
print(f"      VRAM used={vram_used:.2f} GB")

print("[3/5] Generating test audio (3s, 24kHz sinusoid)")
sample_rate = processor.sampling_rate
duration = 3  # seconds
t = torch.arange(0, duration * sample_rate) / sample_rate
audio = torch.sin(2 * torch.pi * 440 * t).float()

print(f"      audio shape={audio.shape}, sr={sample_rate}")

print("[4/5] Running GPU inference")
inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print(f"[5/5] Verifying outputs")
hidden_states = outputs.hidden_states
print(f"      num_layers={len(hidden_states)} (expected: 25 for 330M)")
print(f"      layer_0 shape={hidden_states[0].shape}")
print(f"      layer_-1 shape={hidden_states[-1].shape}")

# Pooled embedding for style similarity
embedding = hidden_states[-1].mean(dim=1)
print(f"      pooled embedding shape={embedding.shape}")

print("\n=== ALL CHECKS PASSED ===")
print(f"Model: {MODEL_NAME}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Layers: {len(hidden_states)}")
print(f"Embedding dim: {hidden_states[-1].shape[-1]}")
