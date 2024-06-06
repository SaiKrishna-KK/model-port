# ModelPort 🧳
ModelPort lets you export and run your machine learning models **anywhere** — regardless of architecture or operating system.

## ✅ Features
- Export PyTorch models to ONNX
- Package model + inference code
- Run cross-platform with Docker (`x86_64`, `arm64`, etc.)

## 📦 Getting Started

```bash
# Export your model
python modelport.py export model.pt

# Run it on desired platform
python modelport.py run modelport_export --arch linux/arm64
```

## 🔧 Architecture Support
- ✅ x86_64 (Intel, AMD)
- ✅ arm64 (Apple M1/M2, Jetson, Raspberry Pi) 