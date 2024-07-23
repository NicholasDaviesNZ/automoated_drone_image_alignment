from PIL import Image, ImageOps
from create_datasets import pad_image, combined_affine_transform

base_image_path = "/workspaces/automoated_drone_image_alignment/test_training_set_images/darfield radish 03042023_georeferenced.png"
padded_image_size = (1280, 1280)
output_res = (1024, 1024)



# Open and resize the image
img = Image.open(base_image_path).convert('RGB')
img_resize = img.resize(output_res, Image.LANCZOS)
img_transformed, affine_matrix = combined_affine_transform(img_resize, padded_size=padded_image_size)

# Transform the image back
transformed_back_image = img_transformed.transform(
    img_transformed.size,
    Image.AFFINE,
    affine_matrix,
    resample=Image.BILINEAR
)

# Calculate crop box
pad_width = padded_image_size[0] - output_res[0]
pad_height = padded_image_size[1] - output_res[1]
pad_left = pad_width // 2
pad_top = pad_height // 2
pad_right = pad_width - pad_left
pad_bottom = pad_height - pad_top
crop_box = (pad_left, pad_top, padded_image_size[0] - pad_right, padded_image_size[1] - pad_bottom)

# Crop and resize back to original size
unpadded_img = transformed_back_image.crop(crop_box)

rebuilt_transformed_image = unpadded_img.resize(img.size, Image.LANCZOS)



