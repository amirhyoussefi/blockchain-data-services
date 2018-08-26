# -*- coding: utf-8 -*-
import urllib2
import json, requests
from random import randint
import os
from flask import Flask, flash, request, redirect, url_for, render_template, current_app, session
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:/Python27/BitClaveDestination/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/uploadFile", methods=['GET', 'POST'])
def uploaded_file():
    if request.method == 'GET':        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))    
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
@app.route("/", methods=['GET', 'POST'])
def landing_page():
    return app.send_static_file('demo.html')

if __name__ == "__main__":
    app.debug = True
    app.run()