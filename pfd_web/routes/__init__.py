from .home import home_bp
from .video import video_bp
from .test import test_bp
from .scene import scene_bp
from .opt import opt_bp
from .data import data_bp


def init_app(app):
    print("routes init")
    app.register_blueprint(home_bp)
    app.register_blueprint(test_bp, url_prefix='/test')
    app.register_blueprint(video_bp, url_prefix='/video')
    app.register_blueprint(scene_bp, url_prefix='/scene')
    app.register_blueprint(opt_bp, url_prefix='/opt')
    app.register_blueprint(data_bp, url_prefix='/data')
