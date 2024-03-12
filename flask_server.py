from flask import Flask,request,render_template,jsonify
from test_deepeval import *
import json 
import os

app = Flask(__name__)

# Replace with your path
json_file_path = "C:/Training Exercices/Trace_Deepeval_llamaindex/Exercices/temp_test_run_data.json"

@app.route('/info', methods=['POST'])
def get_json_content():
    chat = Chatbot()
    output_data=test_hallucination(chat)

    print('Token',chat.total_tokens)
    with open(json_file_path, "r") as file:
        json_content = json.load(file)
    
    response_data = {
        **json_content,
        'Token Usage': chat.total_tokens
    }
    
    return jsonify(response_data)

# print(chat.total_tokens)
# @app.route('/plateforme',methods=['POST'])
# def index():

#     chat= Chatbot()
#     output_data=test_hallucination(chat)
#     print('done !')
#     return output_data
# output_data =index()
@app.route('/test')
def test():

    return render_template('dashboard.html', output_data=output_data)
if __name__ == '__main__':
    app.run(debug=True)

    