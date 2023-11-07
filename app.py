import os
import json

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)
from flask_cors import CORS
import logging

import gen_ai


app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

# route for healthcheck
@app.route('/healthcheck', methods=["GET"])
def healthcheck():
    # Returning an api for showing in reactjs
    return {"status": "OK"}
# route for chat
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    logging.info("API request param:", data)
    question = data["question"]
    json_response = gen_ai.llm_pipeline_with_history(question)
    
    # Convert the JSON response to a JSON-serializable format    
    # Return the JSON response
    return jsonify(json_response)

if __name__ == '__main__':
   app.run()
