#!/usr/bin/env python

import os
import sys
import time
import signal

import numpy as np
import pandas as pd

import torch
import aiohttp
import asyncio
import uvicorn

from fastai import *
from fastai.tabular import *
from fastai.tabular.learner import TabularLearner
from fastai.learner import load_learner

import hashlib

import shutil

from io import BytesIO
from io import StringIO

import binascii

from flask import Flask, request
from flask_debugtoolbar import DebugToolbarExtension
import logging

from wwf.tab.export import *

from threading import Thread

from colored import fg, bg, attr

import kistlerfile
from kistlerfile import KistlerFile
from kistlerfile import create_inference_ready_sample

import socket

import argparse

from argument_parser import define_boolean_argument
from argument_parser import var2opt

from starlette_auth  import *
from bcrypt_password import user_pass_db, create_user_pass_db

classes = [False, True]		# for the ML model
basepath = None
userpath = None

def set_paths():
	hostname=socket.gethostname()
	if hostname == 'zapp-brannigan':	# my PC, let's test stuff in /tmp please
		basepath = Path('/tmp/static')
		userpath = Path('/tmp/userdata')
	else:
		basepath = Path('/app/static')
		userpath = Path('/app/userdata')
	return basepath, userpath


# --------------------------------------------------
# ============== ROUTES AND ENDPOINTS ==============
# --------------------------------------------------
@requires("authenticated", redirect="homepage")
async def admin(request):
	return JSONResponse(
	{
		"authenticated": request.user.is_authenticated,
		"user": request.user.display_name,
	}
	)
@requires("authenticated", redirect="login")
async def userdata(request, debug=False):
	# e.g. http://deeplearning.ge.imati.cnr.it:55514/userdata/graph/Part_maXYmos7_MP-001_2021-05-25_10-30-23_2_2105250957185363_______OK.png
	if debug:
		return JSONResponse(
		{
			"authenticated": request.user.is_authenticated,
			"user": request.user.display_name,
			"path": request.path_params.rest_of_path,
		})
	else:
		print(f'Received request: {request.path_params}')
		extra_path = request.path_params["rest_of_path"]
		print(f'Received path   : {extra_path}')
		if 'html' in Path(extra_path).suffix:
			# e.g. http://deeplearning.ge.imati.cnr.it:55514/userdata/html/Part_maXYmos7_MP-001_2021-05-25_10-30-23_2_2105250957185363_______OK.html
			html_file = userpath / extra_path
			return HTMLResponse(html_file.open().read())
		else:
			# e.g. http://deeplearning.ge.imati.cnr.it:55514/userdata/graph/Part_maXYmos7_MP-001_2021-05-25_10-30-23_2_2105250957185363_______OK.png
			#return HTMLResponse(f'<html><body>Resource not allowed: {extra_path}</body></html>')
			generic_file = userpath / extra_path
			return FileResponse(generic_file)

# --------------------------------------------------
# ==================================================
# --------------------------------------------------

basepath, userpath = set_paths()

routes = [
		Route("/admin",	endpoint=admin),
		#Route("/test",	endpoint=homepage),
		Route("/",	endpoint=homepage),
		#Route("/login", endpoint=login),
		Route('/userdata/{rest_of_path:path}', userdata)
	]

middleware = [Middleware(AuthenticationMiddleware, backend=BasicAuthBackend())]

app = Starlette(routes=routes, middleware=middleware)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
#app.mount('/static', StaticFiles(directory='app/static'))
app.mount('/static', StaticFiles(directory=basepath))

# --------------------------------------------------
# ==================================================
# --------------------------------------------------

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner(cmd, url, model_name):
	if 'serve' in cmd:
		if url is None or model_name is None:
				message = "\n\nNo model download URL or model destination name has been specified. Exiting...\n"
				raise RuntimeError(message)
	else:
		# do nothing, we're being called by docker's RUN instruction
		return
	print(f'Downloading model from: {url} with model name: {model_name}')
	await download_file(url, path / model_name)
	try:
		learn = load_learner(path / model_name)
		print(f'{learn = }')
		return learn
	except RuntimeError as e:
		if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
			print(e)
			message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
			raise RuntimeError(message)
		else:
			raise

def process_csvdata(csvdata):
	debug = False
	if debug:
		df = pd.read_csv(csvdata, skiprows=KistlerFile.kistler_dataframe_line_offset, sep=';', decimal=',')
		print(f'{df = }')
	kf = KistlerFile(fname=csvdata,debug=False)		# let's see if this trick works...
	if debug:
		prediction = kf.df

	KistlerFile._kf_max_time = 3.9999			# this is mandatory and hardcoded to obtain (exactly) 800 samples after resampling. ML model expects 800 columns.
	kf.resample(debug=False)

	sample = create_inference_ready_sample(kf, debug=False)
	row, clas, probs = learn.predict(sample)
	return row, clas, probs, kf


# --------------------------------------------------
# ====================== HTML ======================
# --------------------------------------------------
@app.route("/login")
async def login(request):
	if request.user.is_authenticated:
		return RedirectResponse(url=f"/", status_code=303)
	else:
		response = Response(headers={"WWW-Authenticate": "Basic"}, status_code=401)
	return response

@app.route('/analyze', methods=['POST'])
async def analyze(request):
	form_data = await request.form()
	starlette_data = await(form_data['file'].read())
	csvdata = StringIO(str(starlette_data.decode("utf-8")))

	row, clas, probs, kf = process_csvdata(csvdata)

	return JSONResponse({'result': str(classes[int(clas)]) + ' -> ' + str(probs)})
# --------------------------------------------------
# ==================================================
# --------------------------------------------------

flask_debug = False
toolbar     = None
flask_app   = Flask(__name__)

# --------------------------------------------------
# ====================== POST ======================
# --------------------------------------------------
@flask_app.route('/debug')
def index():
    logging.warning("See this message in Flask Debug Toolbar!")
    return "<html><body>it works, now try a post request...</body></html>"

@flask_app.route('/post', methods=['POST'])
def result():
	debug = True
	if 'username' in request.form and 'password' in request.form:
		username            = request.form['username']
		plaintext_password  = request.form['password']
	else:
		print(f'\n\n{fg("white")}{bg("red_1")}No username and/or password received, continuing in "compatibility mode" {attr("blink")}(deprecated, will be removed in the future){rst}')
		username            = 'user1'
		plaintext_password  = 'password1'

	password_verified, msg = verify_user_and_password(username, plaintext_password)
	if not password_verified:
		return f"/post received username: {username} -> {msg}{username}"

	print(f'Received filename: {Path(request.form["filename"]).stem}\n\n')
	content = request.form['kistlerfile']
	readable_hash = hashlib.sha256(content.encode('utf-8')).hexdigest();

	csvdata = StringIO(str(content))

	row, clas, probs, kf = process_csvdata(csvdata)
	predstr = str(classes[int(clas)])

	print(f'\n\nPrediction for filename: {Path(request.form["filename"]).stem} -> class: {predstr} -> probs: {str(probs)}\n\n')

	graphfn     = userpath / 'graph' / (Path(request.form["filename"]).stem + '.png')
	resampledfn = userpath / 'graph' / (Path(request.form["filename"]).stem + '-resampled.png')
	csvfn       = userpath / 'data'  / (Path(request.form["filename"]).name)
	htmlfn      = userpath / 'html'  / (Path(request.form["filename"]).stem + '.html')

	print(f'\nWriting graph to: {graphfn}, CSV file to {csvfn} and HTML file to {htmlfn}')

	graphfn.parent.mkdir(exist_ok=True, parents=True)
	csvfn.parent.mkdir(exist_ok=True, parents=True)
	htmlfn.parent.mkdir(exist_ok=True, parents=True)

	kf.graph(resampled=False, both=True, debug=False, filename=graphfn)

	with open(csvfn, 'w') as fd:
		csvdata.seek(0)
		shutil.copyfileobj(csvdata, fd)

	graphfnlink	= Path('..') / Path(*graphfn.parts[3:])		# remove leading '/app/static' (because our relative path is deeplearning.ge.imati.cnr.it:55564/static/html)
	csvfnlink	= Path('..') / Path(*csvfn.parts[3:])		# remove leading '/app/static' (because our relative path is deeplearning.ge.imati.cnr.it:55564/static/html)
	print(f'Writing links: {graphfnlink} - {csvfnlink}')
	template    = open(Path('/') / basepath / 'template.html').read().format(csv_fn=csvfnlink, graph_fn=graphfnlink, pred=predstr)

	with open(htmlfn, "w") as html_file:
		html_file.write(template)

	return f"/post received: {request.form['filename']} -> sha256: {readable_hash} -> class: {predstr} -> probs: {str(probs)}"
# --------------------------------------------------
# ==================================================
# --------------------------------------------------

def load_pandas(fname):
	"Load in a `TabularPandas` object from `fname`"
	distrib_barrier()
	res = pickle.load(open(fname, 'rb'))
	return res

def start_flask(port, host='0.0.0.0', debug=False):
	print(f'Running Flask app with {host = }, {port = }')
	if flask_debug:
		app.secret_key = 'asdfasdfqwerqwer'
		toolbar = DebugToolbarExtension(flask_app)
	flask_app.run(host, port, debug)							# POST requests
	return flask_app


def check_port(port):
	if type(port) != int or port < 1 or port > 65535:
		message = "\n\nPlease provide a valid port number. Exiting...\n"
		raise RuntimeError(message)

def argument_parser():
	parser = argparse.ArgumentParser(description='Image Segmentation Inference with Fast.ai v2 and SemTorch')

	parser.add_argument('--cmd',		default=""		, help='the function to execute, default: serve')
	parser.add_argument('--model-name'				, help='the model to load for inference in .pkl format')
	parser.add_argument('--model-url'				, help='the URL where to download the model')
	parser.add_argument('--web-port',	default=55564, type=int	, help='web interface (for debug purposes) port')
	parser.add_argument('--flask-port',	default=55563, type=int	, help='flask TCP port where to receive POST requests')

	args = parser.parse_args()

	print(f'argument_parser() received arguments: {args}')

	return args


args = argument_parser()

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner(args.cmd, args.model_url, args.model_name))]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

if __name__ == '__main__':
	if 'serve' in args.cmd:
		print(f'Starting main python script: {__name__}...')

		user_pass_db = create_user_pass_db()

		host='0.0.0.0'
		port=args.flask_port
		check_port(port)
		t = Thread(target=start_flask, args=(port, host, flask_debug,))
		t.start()

		host='0.0.0.0'
		port=args.web_port
		check_port(port)
		print(f'Creating Uvicorn app with {host = }, {port = }')
		uvicorn.run(app=app, host=host, port=port, log_level="info")			# HTML interface
		print(f'Killing self pid: {os.getpid()} to get rid of Flask...')
		time.sleep(2)
		os.kill(os.getpid(), signal.SIGKILL)

print(f'Main python script: {__name__} reached the end...')

