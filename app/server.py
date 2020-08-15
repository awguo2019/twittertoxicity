import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.text import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import tweepy

export_file_url = 'https://www.googleapis.com/drive/v3/files/1v4rpPSkHkOaYDM1RvExukQMriOqXNbhN?alt=media&key=AIzaSyAZxRmGvTn8CCz084JSla4u5Gei8Af3_VM'
export_file_name = 'export.pkl'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:       
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
   
    
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    
    data = await request.json()
    #data = await request.args['data']
    print("data:", data)
    # img_bytes = await (data['file'].read())
    # took out img_bytes
    # img = open_image(BytesIO(img_bytes))
    
    
    
    img = data["textField"]
    print("data['textField']", data["textField"])
    print("img:", img)
    
    try:
        accesstoken = '1291150581256597504-z0aRzKzhAzMFy5FEeyQaf33VDVYiXZ'
        accesstoken_secret = 'HbZOF0xo3PKio94f3oBNUTSMU09znv4N2oQUGhove9OS6'
        api_key = 'XmgCBVVleESTzLBbesBWNFWRr'
        api_key_secret = 'ikOmADnjBm9W8ujcAMRN5Uzcg3tBwsEtzS2HhtL3NxGiUIEyPn'
        auth = tweepy.OAuthHandler(api_key, api_key_secret)
        auth.set_access_token(accesstoken, accesstoken_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
    except Exception as error:
        return render_template("error in twitter api authentication", error=str(error))

    
    try:
        tweets = tweepy.Cursor(api.search, q=img,lang="en").items(100)
    
        tsum = 0;
        for tweet in tweets:
            print(tweet.text);
            preds, test, tensor = learn.predict(tweet.text)
            tsum = tsum + test.numpy();
    except Exception as error:
        return render_template("error in twitter data", error=str(error))
        
    
    percent = tsum.item();

    #prediction = learn.predict(img)
    #print("prediction:", prediction)
    
    return JSONResponse({'result': percent})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=605413, log_level="info")
