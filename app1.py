import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from dialog import Ui_Dialog
import cv2
import os
import glob
import numpy as np
import cv2
import tensorflow as tf
import time
from mtcnn_master.mtcnn.mtcnn import MTCNN
from facenet_face_recognition_master.fr_utils import *
from facenet_face_recognition_master.inception_blocks_v2 import *
from tensorflow.keras import backend as K
import joblib
import argparse
import pymongo
from datetime import datetime
import json
import pytz

class AppWindow(QDialog):
	def __init__(self):
		super().__init__()
		self.ui = Ui_Dialog()
		self.ui.setupUi(self)

		#prepare face_detection
		self.detector = MTCNN()
		K.set_image_data_format('channels_first')
		self.FRmodel = faceRecoModel(input_shape=(3, 96, 96))

		self.FRmodel.compile(optimizer = 'adam', loss = self.triplet_loss, metrics = ['accuracy'])
		load_weights_from_FaceNet(self.FRmodel)

		#connect database-server
		self.myclient = pymongo.MongoClient("mongodb+srv://VuGiaBao:bao0902429190@cluster0-c4dmj.azure.mongodb.net/face_recognition?retryWrites=true&w=majority")
		self.mydb = self.myclient["Attendance_checking"]
		self.CSDL_col = self.mydb["CSDL"]
		self.Cham_cong_col = self.mydb["Cham_cong"]

		#call database func
		self.data=self.prepare_database()

		# create a timer
		self.timer = QTimer()
		# set timer timeout callback function
		self.timer.timeout.connect(self.recog_pushdata)
		# set control_bt callback clicked  function
		self.ui.Open_bt.clicked.connect(self.controlTimer)

	#create triploss function
	def triplet_loss(self,y_true, y_pred, alpha = 0.3):
		anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
		pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)), axis=-1)
		neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), axis=-1)
		basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
		self.loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
		return self.loss

	#define database
	def prepare_database(self):
		self.database = {}
		for folder in glob.glob("facenet_face_recognition_master/images/*"):
			for file in glob.glob(folder+"/*"):
				self.identity = os.path.splitext(os.path.basename(file))[0]
				self.database[self.identity] = img_path_to_encoding(file, self.FRmodel)
		return self.database

	#define recog function
	def who_is_it(self,image, database, model):
		self.encoding = img_to_encoding(image, model)
		self.min_dist = 100
		self.identity = None

		# Loop over the database dictionary's names and encodings.
		for (name, db_enc) in database.items():
			dist = np.linalg.norm(db_enc - self.encoding)
			if dist < self.min_dist:
				self.min_dist = dist
				self.identity = name

		if self.min_dist > 0.52:
			return None
		else:
			self.identity=self.identity.split("_")[0]
			return self.identity

	def recog_pushdata(self):
		# Capture frame-by-frame
		ret, image = self.cap.read()
		result = self.detector.detect_faces(image)

		for person in result:
			bounding_box = person['box']
			cv2.rectangle(image,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
			crop_img = image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
			id=self.who_is_it(crop_img, self.data, self.FRmodel)

			#push database
			if id == None:
				pass
			else:
				self.ID_found={"ID":id}
				self.res=self.CSDL_col.find_one(self.ID_found,{"_id":0})
				self.res['realtime']= datetime.now(pytz.timezone("Asia/Bangkok"))
				self.Cham_cong_col.insert_one(self.res)

			image=cv2.putText(image,id,(bounding_box[0],bounding_box[1]+bounding_box[3]),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv2.LINE_AA)
			image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# get frame infos
		height, width, channel = image.shape
		step = channel * width
		# create QImage from RGB frame
		qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
		# show frame in img_label
		self.ui.label.setPixmap(QPixmap.fromImage(qImg))

	# start/stop timer
	def controlTimer(self):
		# if timer is stopped
		if not self.timer.isActive():
			# create video capture
			self.cap = cv2.VideoCapture(0)
			# start timer
			self.timer.start(20)
			# update control_bt text
			self.ui.Open_bt.setText("Close")

		# if timer is started
		else:
			# stop timer
			self.timer.stop()
			# release video capture
			self.cap.release()
			# update control_bt text
			self.ui.Open_bt.setText("Open")


app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())

