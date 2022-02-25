import os
from flask import Blueprint, render_template, request, current_app

hui_bp = Blueprint('hui', __name__, template_folder='templates', static_folder='static/hui')


@hui_bp.route('/')
def index():
    # ip = request.remote_addr
    return current_app.send_static_file("hui/index.html")
    return render_template('hui/index.html')