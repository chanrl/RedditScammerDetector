from flask import Flask, request, render_template,jsonify
from joblib import load
from user import *
from ensemble import *

app = Flask(__name__)

#import ensemble classifier
eclf = load('eclf.pkl')

def do_something(text1):
#takes input data from the web form and passes it into the trained ensemble classifier

   proba = eclf.predict(str(text1))
   
   return str(round(proba[0],3))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/join', methods=['GET','POST'])
def my_form_post():
    text1 = request.form['text1']
    word = request.args.get('text1')
    combine = do_something(text1)
    result = {
        "output": combine
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    app.run()