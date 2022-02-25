from flask import Flask
from .settings import BaseConfig


def create_app():
    from . import models, routes, services
    app = Flask(__name__)
    app.config.from_object(BaseConfig)

    # models.init_app(app)
    routes.init_app(app)

    return app

