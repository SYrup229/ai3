import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from tqdm import tqdm

def load_mnist():
    # Define transformations - MNIST is grayscale so we'll repeat it to get 3 channels
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT typically expects 224x224 images
        transforms.Grayscale(3),        # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    return trainloader, testloader

def eval_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST data
    trainloader, testloader = load_mnist()
    
    # Load pretrained ViT model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                     num_labels=10,
                                                     ignore_mismatched_sizes=True)
    model = model.to(device)
    
    # Evaluate the model
    accuracy = eval_model(model, testloader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()