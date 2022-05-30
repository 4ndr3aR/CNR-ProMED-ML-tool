#!/usr/bin/env python3

import base64

from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles

from starlette.authentication import AuthCredentials, AuthenticationBackend, AuthenticationError, SimpleUser, requires, UnauthenticatedUser
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocketDisconnect

from bcrypt_password import rst, encrpyt_password, verify_password, verify_user_and_password, create_user_pass_db

path = Path(__file__).parent

class BasicAuthBackend(AuthenticationBackend):
	async def authenticate(self, conn):
		if "Authorization" not in conn.headers:
			return

		auth = conn.headers["Authorization"]
		try:
			scheme, credentials = auth.split()
			if scheme.lower() != 'basic':
				return
			decoded = base64.b64decode(credentials).decode("ascii")
		except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
			raise AuthenticationError('Invalid basic auth credentials')

		username, _, password = decoded.partition(":")
		# TODO: You'd want to verify the username and password here.
		print(f'{username = } - {password = }')
		password_verified, msg = verify_user_and_password(username, password)
		if not password_verified:
			#return f"BasicAuthBackend received username: {username} -> {msg}{username}"
			return AuthCredentials(), UnauthenticatedUser()
		return AuthCredentials(["authenticated"]), SimpleUser(username)

@requires("authenticated", redirect="login")
async def homepage(request):
	if request.user.is_authenticated:
		html_file = path / 'view' / 'index.html'
		return HTMLResponse(html_file.open().read())
	return PlainTextResponse('Hello, unauthenticated user!')

