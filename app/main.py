from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from middleware.processPrompt import promptResponse
import json

import logging
logging.basicConfig(level=logging.DEBUG)

file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def process_form(
    request: Request, 
    json_file: UploadFile = File(...)
):
    # Read and decode the uploaded JSON file
    try:
        content = await json_file.read()
        json_data = json.loads(content.decode('utf-8'))  # Ensures UTF-8 decoding and JSON parsing
        
        # Format the JSON data for readability
        formatted_json_string = json.dumps(json_data, indent=4, sort_keys=True)
        
        # Await the promptResponse function since it's async
        response_content = await promptResponse(formatted_json_string)
        
        # Format the response for display
        if isinstance(response_content, (dict, list)):
            response_content = json.dumps(response_content, indent=4)
            
    except json.JSONDecodeError:
        response_content = "Invalid JSON file"
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        response_content = f"Error processing request: {str(e)}"
    
    finally:
        # Close the uploaded file after processing
        await json_file.close()

    return templates.TemplateResponse(
        "form.html", 
        {
            "request": request, 
            "response": response_content
        }
    )