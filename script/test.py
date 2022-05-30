#!/usr/bin/env python3
from starlette.applications import Starlette
from starlette.authentication import (
    AuthCredentials, AuthenticationBackend, AuthenticationError, SimpleUser
)
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route
import base64
import binascii


import uvicorn

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
        return AuthCredentials(["authenticated"]), SimpleUser(username)


async def homepage(request):
    if request.user.is_authenticated:
        return PlainTextResponse('Hello, ' + request.user.display_name)
    return PlainTextResponse('Hello, you')

routes = [
    Route("/", endpoint=homepage)
]

middleware = [
    Middleware(AuthenticationMiddleware, backend=BasicAuthBackend())
]

app = Starlette(routes=routes, middleware=middleware)

uvicorn.run(app=app, host='0.0.0.0', port=55514, log_level="info")

