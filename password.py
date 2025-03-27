from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import subprocess
import psutil  # To check if the second app is already running

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

def is_app_running(port):
    """Check if an app is already running on a specific port."""
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = process.info['cmdline']
        if cmdline and f"--port={port}" in " ".join(cmdline):
            return True  # The app is already running
    return False

@app.route('/')
def home():
    if 'user_id' in session:
        return f"Welcome, {session['user_name']}! <a href='/logout'>Logout</a>"
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        if User.query.filter_by(email=email).first():
            return "Email already exists! Try logging in."

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name

            # Check if the second app is already running
            if not is_app_running(5001):
                subprocess.Popen(["python", "app.py"])

            # Redirect to the second Flask app
            return redirect("http://127.0.0.1:5001")  # Adjust the port accordingly
        else:
            return "Invalid credentials, try again!", 401

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
