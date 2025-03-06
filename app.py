from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import aiofiles
import json
import csv
import time
from src.helper import llm_pipeline

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"),name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/upload")
async def chat(request:Request,pdf_file:bytes=File(),filename:str=Form(...)):
    base_folder='static/docs/'
    if not os.path.isdir(base_folder):
        os.makedir(base_folder)
    pdf_filename=os.path.join(base_folder,filename)

    async with aiofiles.open(pdf_filename,'wb') as f:
        await f.write(pdf_file)

    response_data=jsonable_encoder(json.dumps({"msg":'success',"pdf_filename":pdf_filename}))
    res = Response(response_data)
    return res

def make_request(answer_generation_chain,question, retries=5, delay=1):
    for attempt in range(retries):
        try:
            answer = answer_generation_chain.run(question)

            # Check if the response has a status_code attribute (unlikely for LangChain outputs)
            if hasattr(answer, 'status_code') and answer.status_code == 429:
                # print(f"Rate limit hit! Retrying in {delay} seconds...")
                time.sleep(delay)
                # delay *= 2  # Exponential backoff
                continue  # Retry the request

            print("Answer: ", answer)
            print("----------")
            with open("answers.txt", "a") as f:
                f.write(f"Question: {question}\n")
                f.write(f"Answer: {answer}\n")
                f.write("----------\n")
            return answer  # Return the answer if successful

        except Exception as e:
            # print(f"Error occurred: {e}")
            time.sleep(delay)
            # delay *= 2  # Increase delay for next retry

    print("Max retries reached. Skipping this question.")
    return None

def get_csv(file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for question in ques_list:
            if question.strip():  # Avoid empty questions
                print("Question: ", question)
                answer=make_request(answer_generation_chain,question)
                csv_writer.writerow([question, answer])
    return output_file




@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    res = Response(response_data)
    return res



if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8080, reload=True)