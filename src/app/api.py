from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from generators import StarCoder
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi.responses import ORJSONResponse



async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"errors": [exc.detail]}, status_code=exc.status_code)

# app

app = FastAPI(title="Starcoder REST API",
              debug=True,
              default_response_class=ORJSONResponse)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

origins = ['*']


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_exception_handler(HTTPException, http_error_handler)


# define the model
#model_name = "HuggingFaceH4/starchat-alpha"  # Language model that finetuned from StarCoder to act as helpful coding asistant

print("Loading the model")
#code_assistant = SantaCoder(model_name, return_html=False)
code_assistant = StarCoder()

print("Model successfully loaded!")


@app.get("/", response_class=HTMLResponse, tags=["root"])
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse, tags=["predict"])
async def get_assistant_response(request: Request):
    data = await request.form()
    query = data["text"]

    prediction = code_assistant(query)
    return templates.TemplateResponse("prediction.html",
                                      {"request": request, "prediction": prediction, "query": query},)

@app.post("/query", tags=["query"])
async def get_response(request:Request):
    data = await request.json()
    query_str = data["inputs"]
    generation_params = {"max_new_tokens":150}        
    prediction = code_assistant(query_str, parameters=generation_params)
    return JSONResponse(content={"generated_text":prediction})
