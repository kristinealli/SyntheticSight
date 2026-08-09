from PIL import Image

from synthetic_sight.inference import _build_transform


def test_evaluation_transform_shape():
    checkpoint = {
        "image_size": [224, 224],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    transform = _build_transform(checkpoint)
    image = Image.new("RGB", (320, 180), "white")
    tensor = transform(image)
    assert tuple(tensor.shape) == (3, 224, 224)
