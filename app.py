# app.py
from flask import Flask, render_template, request, jsonify
#from model import remove_bg
from model2 import run

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_code', methods=['POST'])
def run_code():
    # Call your Python function here
    #remove_bg()
    run()
    result = "Background removed"
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
