from efficientnet_pytorch.model import EfficientNet
from torchbench.image_classification import ImageNet
import torchvision.transforms as transforms
import PIL

# Define Transforms    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
b0_input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b0'),
    paper_model_name='EfficientNet-B0',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=b0_input_transform,
    batch_size=256,
    num_gpu=1
)

# Define Transforms    
b1_input_transform = transforms.Compose([
    transforms.Resize(273, PIL.Image.BICUBIC),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b1'),
    paper_model_name='EfficientNet-B1',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=b1_input_transform,
    batch_size=256,
    num_gpu=1
)

# Define Transforms    
b2_input_transform = transforms.Compose([
    transforms.Resize(292, PIL.Image.BICUBIC),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b2'),
    paper_model_name='EfficientNet-B2',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=b2_input_transform,
    batch_size=256,
    num_gpu=1
)

# Define Transforms    
b3_input_transform = transforms.Compose([
    transforms.Resize(332, PIL.Image.BICUBIC),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b3'),
    paper_model_name='EfficientNet-B3',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=b3_input_transform,
    batch_size=256,
    num_gpu=1
)

# Efficient Net B4

# Define Transforms    
b4_input_transform = transforms.Compose([
    transforms.Resize(427, PIL.Image.BICUBIC),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b4'),
    paper_model_name='EfficientNet-B4',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=b4_input_transform,
    batch_size=256,
    num_gpu=1
)

"""
# Efficient Net B5

# Define Transforms    
b5_input_transform = transforms.Compose([
    transforms.Resize(332, PIL.Image.BICUBIC),
    transforms.CenterCrop(456),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b5'),
    paper_model_name='EfficientNet-B5',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=b5_input_transform,
    batch_size=256,
    num_gpu=1
)

# Efficient Net B6

# Define Transforms    
b3_input_transform = transforms.Compose([
    transforms.Resize(528, PIL.Image.BICUBIC),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b6'),
    paper_model_name='EfficientNet-B6',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=b6_input_transform,
    batch_size=256,
    num_gpu=1
)

# Efficient Net B7

# Define Transforms    
b7_input_transform = transforms.Compose([
    transforms.Resize(600, PIL.Image.BICUBIC),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b7'),
    paper_model_name='EfficientNet-B7',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=b7_input_transform,
    batch_size=256,
    num_gpu=1
)
"""
