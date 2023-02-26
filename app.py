from flask import Flask,request,render_template,redirect
import sqlite3 as sql
import pandas as pd
import numpy as np
import pickle
import bs4 as bs
import re

from nltk.corpus import stopwords


def review_cleaner(review):
    '''
    Clean and preprocess a review.
    
    1. Remove HTML tags
    2. Use regex to remove all special characters (only keep letters)
    3. Make strings to lower case and tokenize / word split reviews
    4. Remove English stopwords
    5. Rejoin to one string
    '''
    
    #1. Remove HTML tags
    review = bs.BeautifulSoup(review).text
    
    #2. Use regex to find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', review)
    
    #3. Remove punctuation
    review = re.sub("[^a-zA-Z]", " ",review)
    
    #4. Tokenize into words (all lower case)
    review = review.lower().split()
    
    #5. Remove stopwords
    eng_stopwords = set(stopwords.words("english"))
    review = [w for w in review if not w in eng_stopwords]
    
    #6. Join the review to one sentence
    review = ' '.join(review+emoticons)
    # add emoticons to the end

    return(review)

model = pickle.load(open("model.pkl","rb"))
vectorize = pickle.load(open("vector.pkl","rb"))

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html")

@app.route("/data")
def table():
    con = sql.connect("database.db")
    con.row_factory = sql.Row

    cur = con.cursor()
    cur.execute("select * from sent")

    rows = cur.fetchall()
    return render_template("data.html",rows = rows)

@app.route("/", methods = ['POST'])
@app.route("/index", methods = ['POST'])
def results():
    text = request.form['review_text']
    clean_text = review_cleaner(text)
    word_veector = vectorize.transform([clean_text])
    prediction = model.predict(word_veector)[0]

    if prediction == 0:
        feeling = "negative"
    else:
        feeling = "positive"
    prob = model.predict_proba(word_veector).max()

    conn = sql.connect("database.db")
    
    maxid = conn.execute("select max(id) from sent").fetchall()[0][0]

    if not maxid:
        maxid = 0
    maxid+=1

    conn.execute("INSERT INTO sent VALUES (?,?,?)",(maxid,text,feeling))
    conn.commit()
    conn.close()

    return redirect("/sentiment")

@app.route("/sentiment")
def finish():
    con = sql.connect("database.db")
    maxid = con.execute("select max(id) from sent").fetchall()[0][0]

    if not maxid:
        s = ""
    else:
        s = con.execute(f"select prediction from sent where id = {maxid}").fetchall()[0][0]
    return render_template("sentiment.html",value1 = s)

if __name__ == "__main__":
    app.run()