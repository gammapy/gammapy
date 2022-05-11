"""
Serve the docs page and open in default browser.

Combination of these two SO (license CC-BY-SA 4.0) answers:
https://stackoverflow.com/a/51295415/3838691
https://stackoverflow.com/a/52531444/3838691
"""
import time
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

ip = "localhost"
port = 8000
directory = "docs/_build/html"
server_address = (ip, port)
url = f"http://{ip}:{port}"


def open_docs():
    # give http server a little time to startup
    time.sleep(0.1)
    webbrowser.open(url)


if __name__ == "__main__":
    Thread(target=open_docs).start()
    Handler = partial(SimpleHTTPRequestHandler, directory=directory)
    httpd = HTTPServer(server_address, Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
