"""
Copy the relevant lines into your EXISTING app.py.
Do not replace your recognition/attendance routes.
"""

import os
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from backend.auth_routes import router as auth_router

# Your existing app = FastAPI(...) stays as-is.

# Use a long random secret in production:
# PowerShell example:
# $env:ATTENDAI_SESSION_SECRET="replace-with-a-long-random-secret"
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("ATTENDAI_SESSION_SECRET", "CHANGE-THIS-DEVELOPMENT-SECRET"),
    same_site="lax",
    https_only=False,  # Set True when your deployment is HTTPS-only.
    max_age=60 * 60 * 8,
)

# Make your existing templates object available to the auth router.
# If you already have: templates = Jinja2Templates(directory="templates")
app.state.templates = templates

app.include_router(auth_router)


# IMPORTANT:
# If your current dashboard route is "/", move it to "/overview".
#
# Example:
#
# @app.get("/overview")
# async def overview(request: Request):
#     if request.session.get("role") != "admin":
#         return RedirectResponse("/login/admin", status_code=303)
#     ... keep your existing overview logic ...
