from PIL import Image, ImageDraw


def tensor_to_image(tensor):
    """Convert a normalized PyTorch tensor back to a Pillow RGB image."""
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


def save_horizontal_panel(panels, path, label_height=28, label_y=7):
    """Save labeled image tensors as one horizontal comparison figure."""
    images = [(label, tensor_to_image(tensor)) for label, tensor in panels]
    panel_width = images[0][1].width
    panel_height = images[0][1].height
    figure = Image.new(
        "RGB",
        (panel_width * len(images), panel_height + label_height),
        color="white",
    )
    draw = ImageDraw.Draw(figure)

    for index, (label, image) in enumerate(images):
        left = index * panel_width
        figure.paste(image, (left, label_height))
        draw.text((left + 8, label_y), label, fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(path)
