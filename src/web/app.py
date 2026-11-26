from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from PIL import Image
from fastapi.responses import FileResponse

from src.models.extract import extract_image, load_model
from src.search import PlaceIndex


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Загружаем один раз при старте сервера, а не на каждый запрос.
    app.state.index = PlaceIndex.load("artifacts/index/district")
    app.state.model = load_model(
        checkpoint_path="mixvpr_best.pt",
        descriptor_dim=app.state.index.dimension,
    )
    yield


app = FastAPI(lifespan=lifespan)
#app = FastAPI()

@app.post("/api/locate")
async def locate(
    request: Request,
    file: UploadFile = File(...),
    k: int = Query(default=5, ge=1, le=20),
):
    try:
        with Image.open(file.file) as uploaded:
            image = uploaded.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "Не удалось прочитать изображение") from exc
    
    vector = extract_image(request.app.state.model, image)
    matches = request.app.state.index.search(vector, k=k)[0]

    

    return {
        "matches": [
            {
                "rank": match.rank,
                "lat": match.metadata["lat"],
                "lon": match.metadata["lon"],
                "similarity": match.similarity,
                #"photo": photo_data_url(match.metadata["path"]),
            }
            for match in matches
        ]
    }
    """
    return {
        "matches": [
            {
                "rank": 1,
                "lat": 54.7714,
                "lon": 56.0662,
                "similarity": 0.92,
                "photo": "/demo/photo.jpg"
            }
        ]
    }"""

@app.get("/")
def page():
    return FileResponse("src/web/static/index.html")

"""
@app.get("/")
async def root():
    return {"message": "Hello World"}
    """
