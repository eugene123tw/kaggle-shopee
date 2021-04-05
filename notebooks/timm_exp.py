import urllib

import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

model = timm.create_model('efficientnet_b2a', pretrained=True)
model.eval()

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
