from model import inference
from PIL import Image, ImageOps
from create_datasets import pad_image

base_image_path = "/workspaces/automoated_drone_image_alignment/georeferenced_image_pairs/beets_desication/beet_desication_02222024_georeferenced.png"
new_image_path = "/workspaces/automoated_drone_image_alignment/test_image/odm_orthophoto_beet_desication.tif"
padded_image_size=(1280, 1280) 
output_res = (1024,1024)


affine_matrix = inference(base_image_path, new_image_path, padded_image_size=padded_image_size, output_res = output_res, best_model_file = 'best_model.pth')
print(affine_matrix)


new_image = Image.open(new_image_path).convert('RGB')
new_image.save("1.png")
new_image_resize = new_image.resize(output_res, Image.LANCZOS)
new_image_resize.save("2.png")
new_image_padded = pad_image(new_image_resize, padding = padded_image_size)
new_image_padded.save("3.png")
transformed_image = new_image_padded.transform(
        new_image_padded.size,
        Image.AFFINE,
        affine_matrix,
        resample=Image.BILINEAR
    )
transformed_image.save("4.png")
# Calculate crop box
pad_width = padded_image_size[0] - output_res[0]
pad_height = padded_image_size[1] - output_res[1]
pad_left = pad_width // 2
pad_top = pad_height // 2
pad_right = pad_width - pad_left
pad_bottom = pad_height - pad_top
crop_box = (pad_left, pad_top, padded_image_size[0] - pad_right, padded_image_size[1] - pad_bottom)
unpadded_img = transformed_image.crop(crop_box)
unpadded_img.save("5.png")
transformed_unpadded_img = unpadded_img.resize(new_image.size, Image.LANCZOS)

transformed_unpadded_img.save('transformed_image.png')

