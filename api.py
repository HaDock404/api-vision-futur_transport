from fastapi import FastAPI, UploadFile, Response  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from io import BytesIO
import numpy as np  # type: ignore

app = FastAPI()

origins = [
    "http://127.0.0.1",
    "http://localhost",
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


def load_saving_model():
    """
    Load the pre-trained model for image segmentation.

    Returns:
        tensorflow.keras.Model: The loaded model.
    """
    model_path = "./models/model_X.keras"
    model = load_model(model_path, compile=False)
    return model


model = load_saving_model()


def preprocess_image(raw_image):
    """
    Preprocess the raw image for model input.

    Args:
        raw_image (bytes): The raw image bytes.

    Returns:
        numpy.ndarray: The preprocessed image array.
    """
    input_shape = (256, 256, 3)
    image = Image.open(BytesIO(raw_image)).convert('RGB')\
        .resize(input_shape[:2])
    image_array = img_to_array(image)
    image_pil = Image.fromarray(image_array.astype('uint8'))
    image = np.expand_dims(image_pil, axis=0)
    return image


def create_mask(processed_image):
    """
    Generate a segmentation mask for the input image.

    Args:
        processed_image (numpy.ndarray): The preprocessed image array.

    Returns:
        PIL.Image.Image: The segmentation mask.
    """
    predicted_mask = model.predict(processed_image)
    mask1 = predicted_mask[0].copy()
    segmentation_map = np.argmax(mask1, axis=-1)
    tableau_transforme = np.expand_dims(segmentation_map, axis=-1)
    first_mask_array = np.uint8(tableau_transforme)
    transformed_array = np.repeat(first_mask_array, 3, axis=2)
    for x in range(transformed_array.shape[0]):
        for y in range(transformed_array.shape[1]):
            if (transformed_array[x, y] == [0, 0, 0]).all():
                transformed_array[x, y] = [250, 170, 30]
            elif (transformed_array[x, y] == [1, 1, 1]).all():
                transformed_array[x, y] = [0, 0, 142]
            elif (transformed_array[x, y] == [2, 2, 2]).all():
                transformed_array[x, y] = [102, 102, 156]
            elif (transformed_array[x, y] == [3, 3, 3]).all():
                transformed_array[x, y] = [220, 20, 60]
            elif (transformed_array[x, y] == [4, 4, 4]).all():
                transformed_array[x, y] = [153, 153, 153]
            elif (transformed_array[x, y] == [5, 5, 5]).all():
                transformed_array[x, y] = [244, 35, 232]
            elif (transformed_array[x, y] == [6, 6, 6]).all():
                transformed_array[x, y] = [70, 70, 70]
            elif (transformed_array[x, y] == [7, 7, 7]).all():
                transformed_array[x, y] = [70, 130, 180]
    pil_mask = Image.fromarray(transformed_array)
    pil_mask = pil_mask.resize((2048, 1024))
    pil_mask = pil_mask.convert("RGBA")
    return pil_mask


def display(mask, raw_image):
    """
    Overlay the mask on the original image.

    Args:
        mask (PIL.Image.Image): The segmentation mask.
        raw_image (bytes): The raw image bytes.

    Returns:
        bytes: The bytes of the composite image.
    """
    transparency = 128
    mask_data = [(r, g, b, transparency) for r, g, b, _ in mask.getdata()]
    mask.putdata(mask_data)
    image_predicted = Image.open(BytesIO(raw_image))
    image_predicted.paste(mask, (0, 0), mask)

    img_byte_array = BytesIO()
    image_predicted.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()
    return img_byte_array


@app.get("/")
def hello():
    """
    Default route to welcome users and direct them to documentation.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Hi, add /docs to the URL to use the API."}


@app.post("/mask_image")
async def display_mask(file: UploadFile):
    """
    Endpoint to upload an image, generate a segmentation mask,
    and return the result.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        Response: The response containing the masked image.
    """
    raw_image = await file.read()
    processed_image = preprocess_image(raw_image)
    mask = create_mask(processed_image)
    display_img = display(mask, raw_image)
    return Response(content=display_img, media_type="image/png")
