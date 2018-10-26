#!/usr/bin/python
#coding:utf8

from flask import Flask, render_template, url_for, request, redirect, make_response, session, jsonify, json, send_from_directory
import os, Queue, time
from werkzeug import secure_filename
import sys  

reload(sys)
sys.setdefaultencoding('utf8')

app = Flask(__name__, static_url_path='', static_folder='')
app.debug = False
app.config['UPLOAD_FOLDER'] = 'data/'
app.add_url_rule('/', 'root', lambda: app.send_static_file('index.html'))

# @app.route('/data/', methods=['GET', 'POST'])
# def getdata():
# 	return send_from_directory(app.config['UPLOAD_FOLDER'],"chart1.json")

if __name__ == '__main__':
    app.run()