#!/usr/bin/env python

from flask import Flask, request
from flask_debugtoolbar import DebugToolbarExtension
import logging

app = Flask(__name__)

flask_debug = False
toolbar     = None

if flask_debug:
	app.secret_key = 'asdfasdfqwerqwer'
	toolbar = DebugToolbarExtension(app)

@app.route('/')
def index():
    logging.warning("See this message in Flask Debug Toolbar!")
    return "<html><body>it works, now try a post request...</body></html>"

@app.route('/post', methods=['POST'])
def result():
	print(request.form['kistlerfile'])
	return f"Received: {request.form['kistlerfile']}"

app.run(host='0.0.0.0', port=55512, debug=flask_debug)


