from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
from time import strftime
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/api/data', methods=['POST'])
def process_data():
    data = request.json
    # Do something with the data
    return jsonify({'status': 'success', 'data_received': data})


@app.after_request
def after_request(response):
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    logger.error('%s %s %s %s %s %s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, response.status)
    return response

@app.errorhandler(Exception)
def exceptions(e):
    tb = traceback.format_exc()
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, tb)
    return e.status_code

if __name__ == '__main__':
    handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=3)
    logger = logging.getLogger('tdm')
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    
    app.run(host='0.0.0.0', port=8000)