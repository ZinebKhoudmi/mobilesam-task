import os

import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image
from fastapi import FastAPI, UploadFile
from io import BytesIO
from fastapi.responses import Response
from fastapi.exceptions import HTTPException

from tools import box_prompt, format_results, point_prompt, fast_process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_checkpoint = "./mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)

@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global mask_generator

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )
    return fig


app = FastAPI()


@app.post("/segment-image/")
async def uploadfile(file: UploadFile):
    # check the content type (MIME type)
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Read the contents of the uploaded file
    file_contents = await file.read()
    # Convert the file contents to a PIL Image object
    image = Image.open(BytesIO(file_contents))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    fig = segment_everything(
        image=image
    )
    with BytesIO() as buf:
        if fig.mode == 'RGBA':
            fig = fig.convert('RGB')
        fig.save(buf, format='PNG')
        img_bytes = buf.getvalue()
    return Response(img_bytes, media_type="image/png")
