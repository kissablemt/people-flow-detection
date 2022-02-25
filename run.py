import sys, os
import pfd_web
import tensorboard as tb
from werkzeug import serving
from werkzeug.middleware import dispatcher

pd_path = os.path.join(os.path.realpath('.'), 'person_detection')
sys.path.append(pd_path)

HOST = '0.0.0.0'
PORT = 5000
LOGDIR = os.path.join(pd_path, 'output')

flask_app = pfd_web.create_app()


class CustomServer(tb.program.TensorBoardServer):
    def __init__(self, tensorboard_app, flags):
        del flags
        self._app = dispatcher.DispatcherMiddleware(
            flask_app, {"/tensorboard": tensorboard_app}
        )

    def serve_forever(self):
        serving.run_simple(HOST, PORT, self._app, use_reloader=True, use_debugger=True, )

    def get_url(self):
        return "http://%s:%s" % (HOST, PORT)

    def print_serving_message(self):
        pass  # Werkzeug's `serving.run_simple` handles this


def tensorboard_main():
    program = tb.program.TensorBoard(server_class=CustomServer)
    program.configure(logdir=LOGDIR)
    program.main()


if __name__ == '__main__':
    flask_app.run(host=HOST, port=PORT, debug=True)
    # tensorboard_main()
