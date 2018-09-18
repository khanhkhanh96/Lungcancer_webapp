from flask import Flask, request, render_template, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug import secure_filename
import tensorflow as tf
from keras.models import load_model
import os
import csv
from io import StringIO as StringIO
import io
import pandas as pd
from flask import Flask, send_file
import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from flask import Flask, make_response
import SimpleITK as sitk
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data/lungcancer2.db'
db = SQLAlchemy(app)
database = pd.read_csv("data/database2.csv")
global graph,model,model2
graph = tf.get_default_graph()
model = load_model('model/weights.44-0.857.hdf5')
model2 = load_model('model/weights.36-0.920.hdf5')
model._make_predict_function()
model2._make_predict_function()
graph = tf.get_default_graph()

class Patient(db.Model):
    ptid = db.Column(db.String(120), nullable=False, primary_key=True)
    ptname = db.Column(db.String(80),unique = False, nullable=False)
    ptage = db.Column(db.Integer(),unique = False, nullable=False)
    ptgender = db.Column(db.String(),unique = False, nullable=False)


#db.create_all()
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

text1 = """
Patient is in small cell lung cancer
<br>
_____TREATMENT_______________
<br>
If you only have one small tumor in your lung and there is no evidence of cancer in lymph nodes or elsewhere, your doctors may recommend surgery to remove the tumor and the nearby lymph nodes.

Very few patients with SCLC are treated this way. This is only an option if you are in fairly good health and can withstand having all or part of a lung removed.

Before the operation, the lymph nodes in your chest will be checked for cancer spread with mediastinoscopy or other tests, because surgery is unlikely to be a good option if the cancer has spread.

Surgery is generally followed by chemotherapy. If cancer is found in the lymph nodes that were removed, radiation therapy to the chest is usually advised as well. The radiation is often given at the same time as the chemo. Although this increases the side effects of treatment, it appears to be more effective than giving one treatment after the other. You might not be given radiation therapy if you already have severe lung disease (in addition to your cancer) or other serious health problems.

In about half of people with SCLC, the cancer will eventually spread to the brain if no preventive measures are taken. For this reason, you may be given radiation therapy to the head (called prophylactic cranial irradiation, or PCI) to try to prevent this. The radiation is usually given in low doses. Still, some patients may have side effects from the radiation.
<br>___MEDICINE RECOMMEND___________________<br>
- Medicine use for this:<br>
- Cisplatin and etoposide<br>
- Carboplatin and etoposide<br>
- Cisplatin and irinotecan<br>
- Carboplatin and irinotecan"""
text2 = """
Patient is worse than small cell lung cancer
<br>
______TREATMENT__________<br>
For most people with limited stage SCLC, surgery is not an option because the tumor is too large, it’s in a place that can’t be removed easily, or it has spread to nearby lymph nodes or other places in the lung. If you are in good health, the standard treatment is chemo plus radiation to the chest given at the same time (called concurrent chemoradiation). The chemo drugs used are usually etoposide plus either cisplatin or carboplatin.

Concurrent chemoradiation can help people with limited stage SCLC live longer and give them a better chance at cure than giving one treatment (or one treatment at a time). The downside is that this combination has more side effects than either chemo or radiation alone, and it can be hard to take.

People who aren’t healthy enough for chemoradiation are usually treated with chemo by itself. This may be followed by radiation to the chest.

If no measures are taken to prevent it, about half of people with SCLC will have cancer spread to their brain. If your cancer has responded well to initial treatment, you may be given radiation therapy to the head (called prophylactic cranial irradiation, or PCI) to try to prevent this. The radiation is usually given in lower doses than what is used if the cancer had already spread to brain, but some patients may still have side effects from the radiation.

Most people treated with chemo (with or without radiation) for limited stage SCLC will have their tumors shrink significantly. In many, the cancer will shrink to the point where it can no longer be seen on imaging tests. Unfortunately, for most people, the cancer will return at some point.

Because these cancers are hard to cure, clinical trials of newer treatments may be a good option for some people. If you think you might want to take part in a clinical trial, talk to your doctor."""


import matplotlib.animation as animation
def plot_3d(image, id, threshold=-300):
    
    
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    print(image[0].shape)
    image = image.transpose(2,1,0)
    print(image.shape)
    verts, faces, normals, values = measure.marching_cubes_lewiner(image,0)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    def rotate(angle):
        ax.view_init(azim=angle)
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
    rot_animation.save('static/images/'+id+'/rotation.gif', dpi=80, writer='imagemagick')

def recommend(a):
	a = a.reshape(1,32, 32, 1)
	a = a.astype('float32')
	a[a>0] = np.true_divide(a[a>0], 4096)
	a[a<0] = np.true_divide(a[a<0], 1494)
	with graph.as_default():
		cancer = model.predict_classes(a, batch_size = None)

	if cancer == 1:
		stage = model.predict_classes(a, batch_size = None)
		if stage == 0:
			return (text1)
		else:
			return (text2)
	else:
		return ("I think it's not a nodule")
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

@app.route('/checkpatient', methods =  ['POST','GET'])
def checkpatient():
	ptname = request.form.get('ptname')
	ptdob = request.form.get('ptdob')
	datapatient = database[(database['patient name']==ptname) & (database['date of birth']==ptdob)]
	if datapatient.shape[0]==0:
		return redirect(url_for('upload'))
	else:
		return redirect(url_for('showimg',id = datapatient.iloc[0][0]))

	
@app.route('/uploader', methods =  ['POST','GET'])
def uploader():
	target = os.path.join(APP_ROOT,'upload')
	id = ""
	if not os.path.isdir(target):
		os.mkdir(target)
	
	ptname = request.form.get('ptname')
	
	ptage = request.form.get('ptage')
	ptdob = request.form.get('ptdob')
	pthealthy = request.form.get('pththy')
	ptdiseases = request.form.get('ptdis')
	pttohpt = request.form.get('pttohpt')
	print("haha"+pttohpt)
	for file in request.files.getlist("file"):
		print(file) 
		filename = file.filename
		if ".mhd" in filename:
			id = filename.split('.mhd')[0]
		destination = "\\".join([target, filename])
		print(destination)
		file.save(destination)
	ptid = id
	ptfiledir = "data/"+id+".mhd"
	datapatient = [ptid,ptname,ptdob,pthealthy,ptdiseases,pttohpt,ptfiledir]
	with open('data/database2.csv', 'a') as f:
	    writer = csv.writer(f)
	    writer.writerow(datapatient)
	database = pd.read_csv("data/database2.csv")
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
		plt.imshow(img_array[i], cmap=plt.cm.gray)
		plt.savefig(directory+'/%s.png'%i)
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
		plt.imshow(img_array[i], cmap=plt.cm.gray)
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
	data_3d = img_array[z-16:z+16,y-16:y+16,x-16:x+16]
	from skimage.filters import threshold_minimum
	thresh = threshold_minimum(data_3d)
	binary = data_3d > thresh
	# plot_3d(binary,id)
	rcm = recommend(data)
	return render_template('toanalyze.html',rcm = rcm, id = id)




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

