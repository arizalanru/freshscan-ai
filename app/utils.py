import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Define class names for fruit classification (sesuai urutan di dataset)
# Berdasarkan output training: ada ImageLabels.txt yang harus dihapus dari mapping
CLASS_NAMES = [
    'fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum',
    'fresh_orange', 'fresh_tomato', 'stale_apple', 'stale_banana',
    'stale_bitter_gourd', 'stale_capsicum', 'stale_orange', 'stale_tomato'
]

# Class mapping yang benar dari training (tanpa ImageLabels.txt)
TRAINING_CLASS_NAMES = [
    'fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum',
    'fresh_orange', 'fresh_tomato', 'stale_apple', 'stale_banana',
    'stale_bitter_gourd', 'stale_capsicum', 'stale_orange', 'stale_tomato'
]

def load_model(model_path, device):
    """
    Load the trained model from checkpoint
    """
    # Load the checkpoint first to inspect it
    checkpoint = torch.load(model_path, map_location=device)
    
    # Berdasarkan output training, model memiliki 12 classes (tanpa ImageLabels.txt)
    num_classes_in_model = 12  # Sesuai dengan output training yang benar
    
    # Try different model architectures based on the checkpoint structure
    try:
        # Try EfficientNet-B0 first (most likely based on error)
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes_in_model)
        model.load_state_dict(checkpoint)
        print("✅ Loaded EfficientNet-B0 model successfully")
    except:
        try:
            # Try MobileNetV2
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes_in_model)
            model.load_state_dict(checkpoint)
            print("✅ Loaded MobileNetV2 model successfully")
        except:
            try:
                # Try ResNet50 as fallback
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes_in_model)
                model.load_state_dict(checkpoint)
                print("✅ Loaded ResNet50 model successfully")
            except Exception as e:
                raise Exception(f"Could not load any model architecture. Error: {str(e)}")
    
    model.to(device)
    model.eval()
    
    return model

def get_image_transforms():
    """
    Get image preprocessing transforms
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform

def predict_image(model, image, device, transform):
    """
    Make prediction on a single image
    """
    # Preprocess image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Berdasarkan output training, class mapping langsung sesuai dengan CLASS_NAMES
    # Tidak ada ImageLabels.txt di model yang tersimpan
    confidence, predicted = torch.max(probabilities, 0)
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()
    
    # Get top 3 predictions
    top3_prob, top3_indices = torch.topk(probabilities, min(3, len(CLASS_NAMES)))
    top3_predictions = []
    for i in range(len(top3_prob)):
        class_name = CLASS_NAMES[top3_indices[i].item()]
        prob = top3_prob[i].item()
        top3_predictions.append((class_name, prob))
    
    return predicted_class, confidence_score, top3_predictions

def get_fruit_info(predicted_class):
    """
    Get information about the predicted fruit/vegetable
    """
    fruit_info = {
        'fresh_apple': {
            'name': 'Fresh Apple',
            'status': '🍎 Fresh',
            'description': 'This apple looks fresh and good to eat!',
            'color': 'green'
        },
        'fresh_banana': {
            'name': 'Fresh Banana',
            'status': '🍌 Fresh',
            'description': 'This banana is ripe and ready to eat!',
            'color': 'green'
        },
        'fresh_bitter_gourd': {
            'name': 'Fresh Bitter Gourd',
            'status': '🥒 Fresh',
            'description': 'This bitter gourd is fresh and good for cooking!',
            'color': 'green'
        },
        'fresh_capsicum': {
            'name': 'Fresh Capsicum',
            'status': '🫑 Fresh',
            'description': 'This bell pepper is fresh and crispy!',
            'color': 'green'
        },
        'fresh_orange': {
            'name': 'Fresh Orange',
            'status': '🍊 Fresh',
            'description': 'This orange is fresh and juicy!',
            'color': 'green'
        },
        'fresh_tomato': {
            'name': 'Fresh Tomato',
            'status': '🍅 Fresh',
            'description': 'This tomato is fresh and perfect for cooking!',
            'color': 'green'
        },
        'stale_apple': {
            'name': 'Stale Apple',
            'status': '🍎 Stale',
            'description': 'This apple is not fresh. Better not to eat it.',
            'color': 'red'
        },
        'stale_banana': {
            'name': 'Stale Banana',
            'status': '🍌 Stale',
            'description': 'This banana is overripe. Not recommended to eat.',
            'color': 'red'
        },
        'stale_bitter_gourd': {
            'name': 'Stale Bitter Gourd',
            'status': '🥒 Stale',
            'description': 'This bitter gourd is not fresh. Avoid using it.',
            'color': 'red'
        },
        'stale_capsicum': {
            'name': 'Stale Capsicum',
            'status': '🫑 Stale',
            'description': 'This bell pepper is not fresh. Better to discard it.',
            'color': 'red'
        },
        'stale_orange': {
            'name': 'Stale Orange',
            'status': '🍊 Stale',
            'description': 'This orange is not fresh. Not recommended to eat.',
            'color': 'red'
        },
        'stale_tomato': {
            'name': 'Stale Tomato',
            'status': '🍅 Stale',
            'description': 'This tomato is not fresh. Better to discard it.',
            'color': 'red'
        }
    }
    
    return fruit_info.get(predicted_class, {
        'name': 'Unknown',
        'status': '❓ Unknown',
        'description': 'Could not identify the fruit/vegetable.',
        'color': 'gray'
    })
