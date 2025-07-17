from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)

@app.route('/')
def home():
    return "hello world"

@app.route('/username/<username>')
def show_user_profile(username):
    return f"Hello , {username} !"

#rendering templates using Jinja2
@app.route('/avengers/<name>')
def profile(name):
    return render_template('Avengers.html',user=name)

@app.route("/login",methods = ['GET','POST'])
def login():
    if request.method == 'POST':
        return f"welcome , {request.form['username']}"  
    return '''
        <form method="POST">
            <label>Username:</label>
            <input type="text" name="username" required>
            <input type="submit" value="Login">
        </form>
    '''
if __name__ == '__main__':
    app.run(debug=True, port=8000)