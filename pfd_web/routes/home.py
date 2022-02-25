from flask import Blueprint, render_template, request

home_bp = Blueprint('home', __name__)


@home_bp.route('/')
def index():
    ip = request.remote_addr
    return render_template('index.html', ip=ip)

@home_bp.route('/welcome')
def welcome():
    ip = request.remote_addr
    return render_template('welcome.html')


@home_bp.route('/selectROI')
def select_roi():
    return render_template('selectROI.html')
