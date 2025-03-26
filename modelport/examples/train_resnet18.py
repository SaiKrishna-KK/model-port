import torch
import torchvision.models as models

# Create ResNet18 model
model = models.resnet18(pretrained=False)
model.eval()

# Save the model
torch.save(model, "resnet18.pt")
print("✅ ResNet18 model saved to resnet18.pt") 