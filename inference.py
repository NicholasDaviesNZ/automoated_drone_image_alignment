from model import inference
from PIL import Image, ImageOps
from create_datasets import pad_image

base_image_path = "/workspaces/automoated_drone_image_alignment/georeferenced_image_pairs/radish_desication_mats/odm_orthophoto_08032024_georeferenced.png"
new_image_path = "/workspaces/automoated_drone_image_alignment/test_image/odm_orthophoto_radish_desication.tif"
padded_image_size=(1280, 1280) 
output_res = (1024,1024)


affine_matrix = inference(base_image_path, new_image_path, padded_image_size=padded_image_size, output_res = output_res, best_model_file = 'best_model.pth')
print(affine_matrix)


new_image = Image.open(new_image_path).convert('RGB')
new_image_resize = new_image.resize(output_res, Image.LANCZOS)
new_image_padded = pad_image(new_image_resize, padding = padded_image_size)

transformed_image = new_image_padded.transform(
        new_image_padded.size,
        Image.AFFINE,
        affine_matrix,
        resample=Image.BILINEAR
    )

transformed_image = transformed_image.resize(new_image.size, Image.LANCZOS)

transformed_image.save('transformed_image.png')


# the following is just to debug that the image minipulation is working correctly, it is but will leave this here for now
base_image = Image.open(base_image_path).convert('RGB')
base_image_resize = base_image.resize(output_res, Image.LANCZOS)
base_image_padded = pad_image(base_image_resize, padding = padded_image_size)
base_transformed_image = base_image_padded.transform(
        base_image_padded.size,
        Image.AFFINE,
        [1, 0, 0, 0, 1, 0], # affine matrix which does nothing
        resample=Image.BILINEAR
    )

base_transformed_image = base_transformed_image.resize(base_image.size, Image.LANCZOS)
base_transformed_image.save('base_transformed_image.png')