from fastapi import Request
from fastapi.responses import RedirectResponse

def require_admin(request: Request):
    """
    Call at the start of every administrator-only route.
    Returns a redirect response when unauthorized, otherwise None.
    """
    if request.session.get("role") != "admin":
        return RedirectResponse("/login/admin", status_code=303)
    return None

# Usage inside an existing route:
#
# @app.get("/students")
# async def students(request: Request):
#     denied = require_admin(request)
#     if denied:
#         return denied
#     ... existing route logic ...
