from flask import Flask, request, render_template, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug import secure_filename
import os
from io import StringIO as StringIO
import io
from flask import Flask, send_file
import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from flask import Flask, make_response
import SimpleITK as sitk
import math

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data/lungcancer.db'
db = SQLAlchemy(app)


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ptname = db.Column(db.String(80), unique=True, nullable=False)
    ptage = db.Column(db.Integer(), unique=True, nullable=False)
    ptid = db.Column(db.String(120), unique=True, nullable=False)
    ptgender = db.Column(db.String(120), unique=True, nullable=False)


#db.create_all()
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#Define model




# class User(db.Model, UserMixin):
#     id = db.Column(db.Integer, primary_key=True)
#     email = db.Column(db.String(255), unique=True)
#     password = db.Column(db.String(255))
#     active = db.Column(db.Boolean())
#     confirmed_at = db.Column(db.DateTime())
#     roles = db.relationship('Role', secondary=roles_users, backref=db.backref('users', lazy='dynamic'))


# # Setup Flask-Security
# user_datastore = SQLAlchemyUserDatastore(db, User, Role)
# security = Security(app, user_datastore)

#Create a user to test with
# @app.before_first_request
# def create_user():
#     db.create_all()
#     user_datastore.create_user(email='khanh@khanh.com', password='password')
#     db.session.commit()


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/upload')
def upload():
	return render_template('upload.html')
	
@app.route('/uploader', methods =  ['POST','GET'])
def uploader():
	target = os.path.join(APP_ROOT,'upload')
	id = ""
	if not os.path.isdir(target):
		os.mkdir(target)
	for file in request.files.getlist("file"):
		print(file) 
		filename = file.filename
		if ".mhd" in filename:
			id = filename.split('.mhd')[0]
		destination = "\\".join([target, filename])
		print(destination)
		file.save(destination)
	ptname = request.form.get('ptname')
	ptid = id
	ptage = request.form.get('ptage')
	ptgender = request.form.get('ptgen')
	patient = Patient(ptname=ptname, ptage = ptage, ptgender = ptgender, ptid = ptid)
	db.session.add(patient)
	db.session.commit()
	return render_template('uploaddone.html',id = id)



@app.route('/plot/<id>')
def plot(id):
	itk_img = sitk.ReadImage('upload/'+id+'.mhd')
	img_array = sitk.GetArrayFromImage(itk_img)
	print("img_array.shape = ", img_array.shape)
	n_images = img_array.shape[0]
	directory = 'static/images/'+id
	if not os.path.exists(directory):
		os.makedirs(directory)
		print("created destination	")
	for i in range(0, n_images):
		fig = plt.figure(frameon=False,figsize=(5.12,5.12),facecolor = 'red', dpi=100)
		ax = fig.add_axes([0, 0, 1, 1])
		ax.axis('off')
		plt.imshow(img_array[3], cmap=plt.cm.gray)
		plt.savefig(directory+'/%s.png'%i)
	return redirect(url_for('showimg',id = id))


@app.route('/showimg/<id>')
def showimg(id):
	user = Patient.query.filter_by(ptid=id).first()
	hists = os.listdir('static/images/'+id)
	return render_template('show.html',hists = hists,id = id, user = user)

@app.route('/showone/<id>/<oneimg>')
def showoneimg(oneimg,id):
	
	return render_template('showone.html',oneimg = oneimg, id = id)

@app.route('/showone/<id>/<oneimg>/submit', methods=['GET','POST'])
def gensubimg(id, oneimg):
	x = request.form.get('x')
	x = int(x)
	y = request.form.get('y')
	y = int(y)
	z = oneimg.split('.')[0]
	z = int(z)
	fileuse = id + ".mhd"
	mhd_file = "upload/"+fileuse
	itk_img = sitk.ReadImage(mhd_file) 
	img_array = sitk.GetArrayFromImage(itk_img)  # z,y,x ordering
	data = img_array[z,y-16:y+16,x-16:x+16]

	from keras.models import load_model
	modelbest = load_model('model/weights.32-0.902.hdf5')
	test = data.reshape(1, 32, 32, 1)
	classes = modelbest.predict_classes(test, batch_size = None)
	if classes == 0:
		return('0')
	if classes == 1:
		return ("1")




if __name__ == "__main__":
    app.run()

# @app.route('/test')
# def profile():
	
# 	return render_template('test.html')

# @app.route('/post_user', methods=['POST'])
# def post_user():
# 	user = User(request.form['username'], request.form['email'])
# 	db.session.add(user)
# 	db.session.commit()
# 	return redirect(url_for('index'))

# @app.route('/modal/<modal>')
# def modal(modal):
# 	return render_template('modal.html',modal=modal)

