from fastapi import APIRouter,Request,Form
from fastapi.responses import HTMLResponse,RedirectResponse
import secrets
import string
from backend.auth_database import (
 initialize_auth_database,authenticate_admin,authenticate_student,
 change_student_password,get_student_attendance,
 change_admin_password,reset_student_password
)
router=APIRouter()
initialize_auth_database()

@router.get("/",response_class=HTMLResponse)
async def access_portal(request:Request):
    if request.session.get("role")=="admin": return RedirectResponse("/overview",303)
    if request.session.get("role")=="student":
        target="/student/change-password" if request.session.get("must_change_password",False) else "/student/dashboard"
        return RedirectResponse(target,303)
    return request.app.state.templates.TemplateResponse(request=request,name="access.html",context={})

@router.get("/login/admin",response_class=HTMLResponse)
async def admin_login_page(request:Request):
    return request.app.state.templates.TemplateResponse(request=request,name="admin_login.html",context={"error":None})

@router.post("/login/admin",response_class=HTMLResponse)
async def admin_login(request:Request,username:str=Form(...),password:str=Form(...)):
    a=authenticate_admin(username,password)
    if a:
        request.session.clear();request.session["role"]="admin";request.session["username"]=a["username"]
        return RedirectResponse("/overview",303)
    return request.app.state.templates.TemplateResponse(request=request,name="admin_login.html",
        context={"error":"Invalid administrator credentials."},status_code=401)

@router.get("/login/student",response_class=HTMLResponse)
async def student_login_page(request:Request):
    return request.app.state.templates.TemplateResponse(request=request,name="student_login.html",context={"error":None})

@router.post("/login/student",response_class=HTMLResponse)
async def student_login(request:Request,username:str=Form(...),password:str=Form(...)):
    a=authenticate_student(username,password)
    if a:
        request.session.clear()
        request.session["role"]="student";request.session["roll_no"]=a["roll_no"]
        request.session["student_name"]=a["name"]
        request.session["must_change_password"]=a["must_change_password"]
        return RedirectResponse("/student/change-password" if a["must_change_password"] else "/student/dashboard",303)
    return request.app.state.templates.TemplateResponse(request=request,name="student_login.html",
        context={"error":"Invalid roll number or password."},status_code=401)

@router.get("/student/change-password",response_class=HTMLResponse)
async def change_page(request:Request):
    if request.session.get("role")!="student":return RedirectResponse("/login/student",303)
    return request.app.state.templates.TemplateResponse(request=request,name="student_change_password.html",
        context={"error":None,"student_name":request.session.get("student_name","Student"),
        "roll_no":request.session.get("roll_no",""),
        "required_change":bool(request.session.get("must_change_password",False))})

@router.post("/student/change-password",response_class=HTMLResponse)
async def change_password(request:Request,current_password:str=Form(...),
                          new_password:str=Form(...),confirm_password:str=Form(...)):
    if request.session.get("role")!="student":return RedirectResponse("/login/student",303)
    ctx={"error":None,"student_name":request.session.get("student_name","Student"),
         "roll_no":request.session.get("roll_no",""),
         "required_change":bool(request.session.get("must_change_password",False))}
    if new_password!=confirm_password:ctx["error"]="New password and confirmation do not match."
    elif len(new_password)<8:ctx["error"]="New password must contain at least 8 characters."
    if ctx["error"]:
        return request.app.state.templates.TemplateResponse(request=request,name="student_change_password.html",context=ctx,status_code=400)
    try: ok=change_student_password(ctx["roll_no"],current_password,new_password)
    except ValueError as e:
        ctx["error"]=str(e)
        return request.app.state.templates.TemplateResponse(request=request,name="student_change_password.html",context=ctx,status_code=400)
    if not ok:
        ctx["error"]="Current password is incorrect."
        return request.app.state.templates.TemplateResponse(request=request,name="student_change_password.html",context=ctx,status_code=401)
    request.session["must_change_password"]=False
    return RedirectResponse("/student/dashboard?password_changed=1",303)

@router.get("/student/dashboard",response_class=HTMLResponse)
async def student_dashboard(request:Request):
    if request.session.get("role")!="student":return RedirectResponse("/login/student",303)
    if request.session.get("must_change_password",False):return RedirectResponse("/student/change-password",303)
    r=request.session.get("roll_no","");records=get_student_attendance(r)
    present=sum(1 for x in records if str(x.get("status","")).strip().lower()=="present")
    total=len(records);rate=round(present/total*100,1) if total else 0
    return request.app.state.templates.TemplateResponse(request=request,name="student_dashboard.html",
        context={"student":{"name":request.session.get("student_name","Student"),"roll_no":r},
        "attendance_records":records,"total_records":total,"present_days":present,
        "attendance_rate":rate,"password_changed":request.query_params.get("password_changed")=="1"})

@router.get("/logout")
async def logout(request:Request):
    request.session.clear();return RedirectResponse("/",303)


def _require_admin(request: Request):
    if request.session.get("role") != "admin":
        return RedirectResponse(
            "/login/admin",
            status_code=303,
        )
    return None


def _temporary_password(length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits + "@#$!"

    while True:
        password = "".join(
            secrets.choice(alphabet)
            for _ in range(length)
        )

        if (
            any(ch.islower() for ch in password)
            and any(ch.isupper() for ch in password)
            and any(ch.isdigit() for ch in password)
            and any(ch in "@#$!" for ch in password)
        ):
            return password







@router.post(
    "/admin/reset-student-password",
    response_class=HTMLResponse,
)
async def admin_reset_student_password_route(
    request: Request,
    roll_no: str = Form(...),
):
    redirect = _require_admin(request)
    if redirect:
        return redirect

    context = {
        "error": None,
        "success": None,
        "reset_credentials": None,
        "admin_username": request.session.get(
            "username",
            "Administrator",
        ),
    }

    temporary_password = _temporary_password()

    try:
        student = reset_student_password(
            roll_no,
            temporary_password,
        )
    except ValueError as exc:
        context["error"] = str(exc)
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="admin_security.html",
            context=context,
            status_code=400,
        )

    context["success"] = (
        "Student password reset successfully. "
        "Share the temporary password securely."
    )
    context["reset_credentials"] = {
        "name": student["name"],
        "roll_no": student["roll_no"],
        "username": student["roll_no"],
        "temporary_password": temporary_password,
    }

    return request.app.state.templates.TemplateResponse(
        request=request,
        name="admin_security.html",
        context=context,
    )


def _admin_required(request: Request):
    if request.session.get("role") != "admin":
        return RedirectResponse(
            url="/login/admin",
            status_code=303,
        )
    return None


@router.get(
    "/admin/security",
    response_class=HTMLResponse,
)
async def administrator_security_page(
    request: Request,
):
    redirect = _admin_required(request)
    if redirect:
        return redirect

    return request.app.state.templates.TemplateResponse(
        request=request,
        name="admin_security.html",
        context={
            "error": None,
            "success": None,
            "admin_username": request.session.get(
                "username",
                "Administrator",
            ),
        },
    )


@router.post(
    "/admin/change-password",
    response_class=HTMLResponse,
)
async def administrator_change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
):
    redirect = _admin_required(request)
    if redirect:
        return redirect

    context = {
        "error": None,
        "success": None,
        "admin_username": request.session.get(
            "username",
            "Administrator",
        ),
    }

    if new_password != confirm_password:
        context["error"] = (
            "New password and confirmation do not match."
        )
    elif len(new_password) < 8:
        context["error"] = (
            "New password must contain at least 8 characters."
        )

    if context["error"]:
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="admin_security.html",
            context=context,
            status_code=400,
        )

    try:
        changed = change_admin_password(
            request.session.get("username", ""),
            current_password,
            new_password,
        )
    except ValueError as exc:
        context["error"] = str(exc)
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="admin_security.html",
            context=context,
            status_code=400,
        )

    if not changed:
        context["error"] = (
            "Current administrator password is incorrect."
        )
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="admin_security.html",
            context=context,
            status_code=401,
        )

    context["success"] = (
        "Administrator password changed successfully."
    )

    return request.app.state.templates.TemplateResponse(
        request=request,
        name="admin_security.html",
        context=context,
    )


@router.get(
    "/admin/student-passwords",
    response_class=HTMLResponse,
)
async def student_password_management_page(
    request: Request,
):
    redirect = _admin_required(request)
    if redirect:
        return redirect

    error = request.session.pop(
        "student_password_management_error",
        None,
    )

    return request.app.state.templates.TemplateResponse(
        request=request,
        name="student_password_management.html",
        context={
            "error": error,
        },
    )


@router.post(
    "/admin/student-passwords/generate",
    response_class=HTMLResponse,
)
async def generate_student_temporary_password(
    request: Request,
    roll_no: str = Form(...),
):
    redirect = _admin_required(request)
    if redirect:
        return redirect

    temporary_password = _temporary_password()

    try:
        student = reset_student_password(
            roll_no,
            temporary_password,
        )
    except ValueError as exc:
        request.session[
            "student_password_management_error"
        ] = str(exc)

        return RedirectResponse(
            url="/admin/student-passwords",
            status_code=303,
        )

    request.session["temporary_student_credentials"] = {
        "name": student["name"],
        "roll_no": student["roll_no"],
        "username": student["roll_no"],
        "temporary_password": temporary_password,
    }

    return RedirectResponse(
        url="/admin/student-passwords/result",
        status_code=303,
    )


@router.get(
    "/admin/student-passwords/result",
    response_class=HTMLResponse,
)
async def student_temporary_password_result(
    request: Request,
):
    redirect = _admin_required(request)
    if redirect:
        return redirect

    credentials = request.session.pop(
        "temporary_student_credentials",
        None,
    )

    if credentials is None:
        return RedirectResponse(
            url="/admin/student-passwords",
            status_code=303,
        )

    return request.app.state.templates.TemplateResponse(
        request=request,
        name="student_password_result.html",
        context={
            "credentials": credentials,
        },
    )
