from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from img_compress import ImagePCACompressor,np

app = FastAPI()

#now Solved



@app.post("/compress-image/")
async def compress_image(
    file: UploadFile = File(...),
    variance: float = Form(..., description="Variance ratio to keep (0-1)")
):
    try:
        # Read uploaded image
        original_bytes = await file.read()
        original_size = len(original_bytes)

        img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        img_array = np.array(img)

        # Compress image
        compressor = ImagePCACompressor(variance_ratio=variance)
        compressed_array = compressor.fit_transform(img_array)

        # Save compressed image to memory
        output_buffer = io.BytesIO()
        Image.fromarray(compressed_array).save(output_buffer, format="JPEG")
        compressed_bytes = output_buffer.getvalue()

        # Stats
        stats = compressor.compression_stats(original_size, len(compressed_bytes))

        # Encode compressed image as base64
        compressed_base64 = base64.b64encode(compressed_bytes).decode("utf-8")

        return {
            "stats": stats,
            "compressed_image_base64": compressed_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()