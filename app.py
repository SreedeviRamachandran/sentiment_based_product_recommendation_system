from flask import Flask, render_template, request
import pandas as pd
import model

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def home():    
    if request.method == 'POST':
        user_name = request.form['userInput']
        display_details = model.top_20_user_recommendations(user_name)
        return render_template("index.html", placeholder_text=display_details)
    if request.method == 'GET':
        return render_template("index.html", placeholder_text=pd.DataFrame())

if __name__ == '__main__':
    app.run()