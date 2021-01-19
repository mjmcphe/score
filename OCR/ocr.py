import cv2
import numpy as np
import os
import shutil
import sys
from twisted.python import log
from twisted.internet import reactor
from autobahn.twisted.websocket import WebSocketServerProtocol, WebSocketServerFactory, listenWS
from twisted.web.server import Site
from twisted.web.static import File
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import qimage2ndarray
import json
import socket


_application_path = os.getcwd()

def get_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
	return cv2.medianBlur(image,5)

def thresholding(image, low, high):
	return cv2.threshold(image, low, high, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def dilate(image):
	kernel = np.ones((5,5),np.uint8)
	return cv2.dilate(image, kernel, iterations = 1)

def erode(image):
	kernel = np.ones((2,2),np.uint8)
	return cv2.erode(image, kernel, iterations = 2)

def opening(image):
	kernel = np.ones((5,5),np.uint8)
	return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
	return cv2.Canny(image, 100, 200)

def min_max(value, min, max):
	if value < min:
		return min
	elif value > max:
		return max
	else:
		return value

def autocrop(image, threshold=0):
	if len(image.shape) == 3:
		flatImage = np.max(image, 2)
	else:
		flatImage = image
	assert len(flatImage.shape) == 2

	rows = np.where(np.max(flatImage, 0) > threshold)[0]
	if rows.size:
		cols = np.where(np.max(flatImage, 1) > threshold)[0]
		image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
	else:
		image = image[:1, :1]

	return image

def clock_to_sec(clock):
	if ":" in clock:
		if clock.split(":")[0] != "" and clock.split(":")[1] != "":
			return int(clock.split(":")[0]) * 60 + int(clock.split(":")[1])
		else:
			return 0
	elif "." in clock:
		if clock.split(".")[0] != "" and clock.split(".")[1] != "":
			return int(clock.split(".")[0])
		else:
			return 0
	else:
		return 0


class WebSocketsWorker(QtCore.QThread):
	updateProgress = QtCore.Signal(list)
	error = QtCore.Signal(str)
	socket_opened = QtCore.Signal(int)

	class BroadcastServerProtocol(WebSocketServerProtocol):
		def onOpen(self):
			self.factory.register(self)

		def onMessage(self, payload, isBinary):
			if not isBinary:
				msg = "{}".format(payload.decode('utf8'))
				self.factory.broadcast(msg)

		def connectionLost(self, reason):
			WebSocketServerProtocol.connectionLost(self, reason)
			self.factory.unregister(self)

	class BroadcastServerFactory(WebSocketServerFactory):
		def __init__(self, url, debug=False, debugCodePaths=False):
			WebSocketServerFactory.__init__(self, url, debug, debugCodePaths)

			self.clients = []
			self.tickcount = 0
			#self.tick()

		def tick(self):
			self.tickcount += 1
			self.broadcast("tick %d from server" % self.tickcount)
			reactor.callLater(0.5, self.tick)

		def register(self, client):
			if client not in self.clients:
				print("registered client {}".format(client.peer))
				self.clients.append(client)

		def unregister(self, client):
			if client in self.clients:
				print("unregistered client {}".format(client.peer))
				self.clients.remove(client)

		def broadcast(self, msg):
			#print("broadcasting message '{}' ..".format(msg))
			for c in self.clients:
				c.sendMessage(msg.encode('utf8'))
				#print("message {} sent to {}".format(msg, c.peer))

		def returnClients(self):
			return
			#for c in self.clients:
				#print(c.peer)


	def __init__(self):
		QtCore.QThread.__init__(self)
		self.factory = self.BroadcastServerFactory("ws://localhost:8000", debug=False, debugCodePaths=False)

	def run(self):
		self.factory.protocol = self.BroadcastServerProtocol
		try:
			listenWS(self.factory)
		except:
			self.error.emit("Fail")
		webdir = File(_application_path + "\\display")
		webdir.indexNames = ['index.php', 'index.html']
		web = Site(webdir)
		try:
			reactor.listenTCP(8080, web)
			self.socket_opened.emit(1)
		except:
			self.error.emit("Fail")
		reactor.run(installSignalHandlers=0)

	def send(self, data):
		reactor.callFromThread(self.factory.broadcast, data)
		self.updateProgress.emit([self.factory.returnClients()])

class ocr_worker(QtCore.QThread):
	def __init__(self, settings):
		QtCore.QThread.__init__(self)

		self.cam_index = settings["cam_index"]
		self.coords = settings["coords"]
		self.warp = {
			"active": -3,
			"done": [False, False, False, False],
			"coords": main.warp,
			"live": [
				[[0,0],[1440,0],[0,720],[1440,720]],
				[[0,0],[1440,0],[0,720],[1440,720]]
			]
		}
		self.res = main.res
		self.pinch = settings["pinch"] #left, top, right, bottom
		self.range = 10
		self.draw_box = {
			"start_coords": [0, 0],
			"current_coords": [0, 0],
			"end_coords": [0, 0],
			"current_digit": -1
		}
		self.pick_color = {
			"coordinates": [0, 0],
			"active": 0
		}
		self.ref_colors = settings["ref_colors"]

		self.colorcount = main.colorcount
		self.temp_font = main.temp_font

		self.font = settings["font"]
		self.decimal_mode = settings["decimal_mode"]
		self.digits = [None] * 10
		self.clock = [0] * 4
		self.score = [0] * 6
		self.clockjson = {
			"method": "clock",
			"data": ""
		}
		self.scorejson = {
			"method": "score",
			"data": {
				"home": {
					"score": ""
				}, "away": {
					"score": ""
				}
			}
		}
		self.last = [0] * 3
		self.rect_colors = [(0,255,255), (0,255,255), (0,255,255), (0,255,255), (0,255,0), (0,255,0), (0,255,0), (0,0,255), (0,0,255), (0,0,255)]

		self.reference_images = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
		self.ref_images_inc = [0] * 10

		self.createmode = -1
		self.existing = 0
		self.send_timeout = False
		self.createdigits = []

		self.load_ref_images()
		self.run()

	def set_warp(self):
		self.warp["done"] = [False, False, False, False]
		for i in range(2):
			self.warp["live"][i] = [[0,0],[self.res[0],0],[0,self.res[1]],[self.res[0],self.res[1]]]

		for index, coord in enumerate(self.warp["coords"]):
			for inner_index, inner_coord in enumerate(coord):
				if abs(0 - inner_coord) <= self.range:
					self.warp["coords"][index][inner_index] = 10
				for direction in self.res:
					if abs(direction - inner_coord) <= self.range:
						self.warp["coords"][index][inner_index] = (direction - 10)

		self.warp["active"] = -2

	def update_warp(self, set):
		if set:
			self.warp["active"] = -3
			self.warp["live"][0] = self.warp["coords"]

		self.warp["live"][1][0] = [0 + self.pinch[0], 0 + self.pinch[1]]
		self.warp["live"][1][1] = [self.res[0] - self.pinch[2], 0 + self.pinch[1]]
		self.warp["live"][1][2] = [0 + self.pinch[0], self.res[1] - self.pinch[3]]
		self.warp["live"][1][3] = [self.res[0] - self.pinch[2], self.res[1] - self.pinch[3]]
		self.warp["done"] = [False, False, False, False]

		if set:
			main.warp = self.warp["coords"]
			main.send()

	def mouse_hover_coordinates(self, event, x, y, flags, param):
		if self.warp["active"] == -1:
			if event == cv2.EVENT_LBUTTONDOWN:
				for index, coord in enumerate(self.warp["coords"]):
					if abs(coord[0] - x) <= self.range and abs(coord[1] - y) <= self.range:
						self.warp["active"] = index

		elif self.warp["active"] > -1:
			if event == cv2.EVENT_MOUSEMOVE:
				self.warp["coords"][self.warp["active"]] = [x, y]

			if event == cv2.EVENT_LBUTTONUP:
				self.warp["done"][self.warp["active"]] = True
				self.warp["active"] = -1

				if all(self.warp["done"]):
					self.update_warp(True)


		if self.draw_box["current_digit"] > -1:
			if event == cv2.EVENT_LBUTTONDOWN:
				self.draw_box["start_coords"] = [x, y]

			if event == cv2.EVENT_MOUSEMOVE:
				self.draw_box["current_coords"] = [x, y]

			if event == cv2.EVENT_LBUTTONUP:
				self.draw_box["end_coords"] = [x, y]

				main.coords[str(self.draw_box["current_digit"])]["x1"] = min(self.draw_box["start_coords"][0], self.draw_box["end_coords"][0])
				main.coords[str(self.draw_box["current_digit"])]["x2"] = max(self.draw_box["start_coords"][0], self.draw_box["end_coords"][0])
				main.coords[str(self.draw_box["current_digit"])]["y1"] = min(self.draw_box["start_coords"][1], self.draw_box["end_coords"][1])
				main.coords[str(self.draw_box["current_digit"])]["y2"] = max(self.draw_box["start_coords"][1], self.draw_box["end_coords"][1])

				for j in ["x", "y"]:
					if main.coords[str(self.draw_box["current_digit"])][str(j) + "1"] == main.coords[str(self.draw_box["current_digit"])][str(j) + "2"]:
						if main.coords[str(self.draw_box["current_digit"])][str(j) + "1"] <= 1:
							main.coords[str(self.draw_box["current_digit"])][str(j) + "1"] = str(int(main.coords[str(self.draw_box["current_digit"])][str(j) + "1"]) - 1)
						else:
							main.coords[str(self.draw_box["current_digit"])][str(j) + "2"] = str(int(main.coords[str(self.draw_box["current_digit"])][str(j) + "2"]) + 1)

				main.update_text_boxes()

				self.draw_box["start_coords"] = [0, 0]
				self.draw_box["end_coords"] = [0, 0]

				main.draw_coords_select = -1

		if self.pick_color["active"] > -1:
			self.pick_color["coords"] = [x, y]


	def load_ref_images(self):
		self.reference_images = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
		self.ref_images_inc = [0] * 10

		for i in range(10):
			self.reference_images[i][0] = cv2.imread(_application_path + "\\frames\\" + self.font + "\\" + str(i) + "_0.png")

		for filename in os.listdir(_application_path + "\\frames\\" + self.font):
			if len(filename) > 2 and "_" in filename:
				self.ref_number = int(filename[0:2].strip("_"))
				self.reference_images[self.ref_number].append(cv2.imread(_application_path + "\\frames\\"  + self.font + "\\" + filename, 0))
				self.ref_images_inc[self.ref_number] += 1

	def check_digit(self, image):
		rnum = ""
		if self.warp["active"] <= -3:
			if self.createmode == 1:
				check = True
				for digit in self.createdigits:
					if digit is not None and digit.shape == image.shape and not(np.bitwise_xor(digit,image).any()):
						check = False
				if check:
					self.createdigits.append(image)
					main.set_clock_text(str(len(self.createdigits) - self.existing) + " Digits Created")
					cv2.imwrite(_application_path + "\\" + self.temp_font + "\\" + str(len(self.createdigits) - self.existing) + ".png", image)
			else:
				for index, ref_digits in enumerate(self.reference_images):
					if ref_digits is not None:
						for ref_digit in ref_digits:
							if ref_digit is not None and ref_digit.shape == image.shape and not(np.bitwise_xor(ref_digit,image).any()):
								rnum = index
		return rnum

	def draw_rect(self, canvas, index, color):
		if self.warp["active"] <= -3:
			if index == self.draw_box["current_digit"] and not (self.draw_box["start_coords"] == [0, 0] and self.draw_box["end_coords"] == [0, 0]):
				text_x_pos = 0
				text_y_pos = 0
				if min(self.draw_box["start_coords"][1], self.draw_box["current_coords"][1]) <= 28:
					text_x_pos = self.draw_box["current_coords"][0]
					text_y_pos = max(self.draw_box["current_coords"][1], self.draw_box["start_coords"][1]) + 28
				else:
					text_x_pos = min(self.draw_box["current_coords"][0], self.draw_box["start_coords"][0])
					text_y_pos = min(self.draw_box["current_coords"][1], self.draw_box["start_coords"][1]) - 5

				cv2.putText(canvas, str(index + 1), (text_x_pos, text_y_pos), cv2.FONT_ITALIC, 1, (color), 2)
				cv2.rectangle(canvas, (self.draw_box["start_coords"][0], self.draw_box["start_coords"][1]), (self.draw_box["current_coords"][0], self.draw_box["current_coords"][1]), color, 1)
			else:
				text_y_pos = 0
				if self.coords[str(index)]["y1"] <= 28:
					text_y_pos = self.coords[str(index)]["y2"] + 28
				else:
					text_y_pos = self.coords[str(index)]["y1"] - 5

				cv2.putText(canvas, str(index + 1), (self.coords[str(index)]["x1"], text_y_pos), cv2.FONT_ITALIC, 1, (color), 2)
				cv2.rectangle(canvas, (self.coords[str(index)]["x1"], self.coords[str(index)]["y1"]), (self.coords[str(index)]["x2"], self.coords[str(index)]["y2"]), (color), 2)

	def run(self):

		cap = cv2.VideoCapture(self.cam_index)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)


		last_cd = -1
		last_clock = "0:00"
		last_score = ""
		last_pinch = []
		timeout_string = ""
		self.frame_counter = 0
		self.end = False

		self.update_warp(True)

		while(cap.isOpened() and not self.end):
			try:
				ret, frame = cap.read()

				if not frame.all(None):
					self.coords = main.check_coords()
					self.ref_colors = main.check_ref_color()
					self.pinch = main.check_pinch()

					self.draw_box["current_digit"] = main.draw_coords_select

					if self.draw_box["current_digit"] != last_cd:
						if self.draw_box["current_digit"] > -1:
							cv2.namedWindow("Update Digit " + str(self.draw_box["current_digit"] + 1))
						last_cd = self.draw_box["current_digit"]

					if main.reload_digits_select:
						self.load_ref_images()
						main.reload_digits_select = False

					if main.draw_warp_select:
						self.set_warp()
						main.draw_warp_select = False

					self.font = main.font

					try:
						temp = main.create_window.get_create_mode()
						if self.createmode < 1 or temp[0] != 0:
							self.createmode, self.font = temp
					except AttributeError:
						self.createmode = -1
						self.font = main.font

					if self.createmode == 0:
						self.load_ref_images()

						try:
						    shutil.rmtree(_application_path + "\\temp")
						except OSError as e:
						    pass

						os.mkdir(_application_path + "\\temp")

						for i in self.reference_images:
							for j in i:
								self.createdigits.append(j)

						self.existing = len(self.createdigits)
						main.ref_images_inc = self.ref_images_inc
						self.createmode = 1

					frame = cv2.resize(frame, (1440, 720), 1, 1, cv2.INTER_NEAREST)
					rows,cols,ch = frame.shape

					pts1 = np.float32(self.warp["live"][0])
					pts2 = np.float32(self.warp["live"][1])
					M = cv2.getPerspectiveTransform(pts1, pts2)
					frame = cv2.warpPerspective(frame,M,(1440,720))

					img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
					rows,cols,_ = img_HSV.shape

					lower_color = []
					upper_color = []
					for i in range(self.colorcount):
						lower_color.append(np.array([((((self.ref_colors[str(i)]["color"][0] - self.ref_colors[str(i)]["range"][0]) / 2)) % 180), (min_max(self.ref_colors[str(i)]["color"][1] - self.ref_colors[str(i)]["range"][1], 0, 255)), (min_max(self.ref_colors[str(i)]["color"][2] - self.ref_colors[str(i)]["range"][2], 0, 255))]))
						upper_color.append(np.array([((((self.ref_colors[str(i)]["color"][0] + self.ref_colors[str(i)]["range"][0]) / 2)) % 180),  (min_max(self.ref_colors[str(i)]["color"][1] + self.ref_colors[str(i)]["range"][1], 0, 255)), (min_max(self.ref_colors[str(i)]["color"][2] + self.ref_colors[str(i)]["range"][2], 0, 255))]))

					img_disp = frame

					if self.colorcount == 1:
						frame = cv2.inRange(img_HSV, lower_color[0], upper_color[0])
					if self.colorcount == 2:
						frame = cv2.bitwise_or(cv2.inRange(img_HSV, lower_color[0], upper_color[0]), cv2.inRange(img_HSV, lower_color[1], upper_color[1]))

					frame = thresholding(frame, 120, 255)
					frame = remove_noise(frame)
					frame = dilate(frame)
					frame = erode(frame)

					if main.show_thresh:
						img_disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

					for i in range(4):
						self.draw_rect(img_disp, i, self.rect_colors[i])
						self.digits[i] = frame[self.coords[str(i)]["y1"]:self.coords[str(i)]["y2"], self.coords[str(i)]["x1"]:self.coords[str(i)]["x2"]]
						self.digits[i] = autocrop(self.digits[i])

						if (cv2.countNonZero(self.digits[i]) == 0):
							if i > 5:
								self.clock[i] = "0"
							else:
								self.clock[i] = ""
						else:
							width = abs(self.coords[str(i)]["x2"] - self.coords[str(i)]["x1"])
							if (self.digits[i].shape[1] < (width // 2)):
								self.digits[i] = cv2.resize(self.digits[i], (1, 9), 1, 1, cv2.INTER_NEAREST)
							elif (self.digits[i].shape[1] >= (width // 2)):
								self.digits[i] = cv2.resize(self.digits[i], (4, 9), 1, 1, cv2.INTER_NEAREST)

							self.digits[i] = thresholding(self.digits[i], 0, 255)

						digit = self.check_digit(self.digits[i])
						if digit != "":
							self.clock[i] = digit

					self.frame_counter += 1
					for i in range(4, 10):
						self.draw_rect(img_disp, i, self.rect_colors[i])

						self.digits[i] = frame[self.coords[str(i)]["y1"]:self.coords[str(i)]["y2"], self.coords[str(i)]["x1"]:self.coords[str(i)]["x2"]]
						self.digits[i] = autocrop(self.digits[i])

						if (cv2.countNonZero(self.digits[i]) == 0):
							self.score[i - 4] = ""
						else:
							if (self.digits[i].shape[1] < (width // 2)):
								self.digits[i] = cv2.resize(self.digits[i], (1, 9), 1, 1, cv2.INTER_NEAREST)
							elif (self.digits[i].shape[1] >= (width // 2)):
								self.digits[i] = cv2.resize(self.digits[i], (4, 9), 1, 1, cv2.INTER_NEAREST)

							self.digits[i] = thresholding(self.digits[i], 0, 255)


						if self.frame_counter % 10 == 0:
							digit = self.check_digit(self.digits[i])
							if digit != "":
								self.score[i - 4] = digit

					if self.warp["active"] == -2:
						cv2.namedWindow("Update Warp")
						self.warp["active"] = -1
					if self.warp["active"] >= -1:
						if cv2.getWindowProperty("Update Warp", 0) == -1:
							self.update_warp(True)
						else:
							cv2.namedWindow("Update Warp")
							cv2.setMouseCallback("Update Warp", self.mouse_hover_coordinates)

							black = np.zeros((720, 1440, 3), np.uint8)
							points = np.array([self.warp["coords"][0],self.warp["coords"][1],self.warp["coords"][3],self.warp["coords"][2]], np.int32)
							points = points.reshape((-1,1,2))
							cv2.fillPoly(black,[points],(255,255,255))
							img_disp = cv2.addWeighted(img_disp, 1.0, black, 0.25, 1)
							cv2.polylines(img_disp,[points],True,(255,255,255), 2)

							for index, coord in enumerate(self.warp["coords"]):
								color = (0, 0, 255)
								if self.warp["active"] == index:
									color = (255, 0, 255)
								elif self.warp["done"][index]:
									color = (255, 0, 0)

								cv2.circle(img_disp, tuple(coord), 3, color, -1)

							cv2.imshow("Update Warp", img_disp)

					else:
						self.update_warp(False)
						cv2.destroyWindow("Update Warp")

					main.display_webcam_feed(img_disp, -1)
					cv2.putText(img_disp, str(self.draw_box["current_coords"][0]) + ", " + str(self.draw_box["current_coords"][1]), (5, 25), cv2.FONT_ITALIC, 1, self.rect_colors[self.draw_box["current_digit"]])

					if self.draw_box["current_digit"] > -1:
						if cv2.getWindowProperty("Update Digit " + str(self.draw_box["current_digit"] + 1), 0) == -1:
							main.draw_coords_select = -1
						else:
							cv2.namedWindow("Update Digit " + str(self.draw_box["current_digit"] + 1))
							cv2.setMouseCallback("Update Digit " + str(self.draw_box["current_digit"] + 1), self.mouse_hover_coordinates)
							cv2.imshow("Update Digit " + str(self.draw_box["current_digit"] + 1), img_disp)
					else:
						for i in range(10):
							cv2.destroyWindow("Update Digit " + str(i + 1))

					self.scorejson["data"]["home"]["score"] = str(self.score[0]) + str(self.score[1]) + str(self.score[2])
					self.scorejson["data"]["away"]["score"] = str(self.score[3]) + str(self.score[4]) + str(self.score[5])

					border_digits = self.digits
					for i in range(10):
						bordersize = 1
						borderleft = 1
						if cv2.countNonZero(border_digits[i]) == 0:
							border_digits[i] = cv2.resize(border_digits[i], (5, 9), 1, 1, cv2.INTER_NEAREST)

						width = abs(self.coords[str(i)]["x2"] - self.coords[str(i)]["x1"])
						if (border_digits[i].shape[1] == 1):
							borderleft = 4

						if i == 7:
							borderleft += 5

						border_digits[i] = cv2.copyMakeBorder(border_digits[i], top=bordersize, bottom=bordersize, left=borderleft, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

					concat_clock = cv2.hconcat([border_digits[0], border_digits[1], border_digits[2], border_digits[3]])
					concat_score = cv2.hconcat([border_digits[4], border_digits[5], border_digits[6], border_digits[7], border_digits[8], border_digits[9]])
					main.display_webcam_feed(concat_clock, 0)
					main.display_webcam_feed(concat_score, 1)
					if self.createmode <= -1:

						if (self.clock[1] != "" and self.clock[2] != "" and self.clock[3] == ""):
							self.clockjson["data"]=(str(self.clock[0]) + str(self.clock[1]) + "." + str(self.clock[2]))
						else:
							self.clockjson["data"]=(str(self.clock[0]) + str(self.clock[1]) + ":" + str(self.clock[2]) + str(self.clock[3]))

						main.set_clock_text(self.clockjson["data"] + timeout_string + " Home " + self.scorejson["data"]["home"]["score"] + " - " + self.scorejson["data"]["away"]["score"] + " Away")

						if last_clock != self.clockjson["data"]:
							last_clock = self.clockjson["data"]

							if (self.clock[0] == "" and self.clock[1] == 1 and self.clock[2] == 0 and self.clock[3] == 0) or (self.clock[0] == "" and self.clock[1] == 0 and self.clock[2] != "" and self.clock[3] != ""):
								timeout_string = " (Timeout)"
								if self.send_timeout:
									main.wsworker.send(json.dumps(self.clockjson))
							else:
								timeout_string = ""
								main.wsworker.send(json.dumps(self.clockjson))

						if last_score != str(self.scorejson["data"]) or self.frame_counter % 90 == 0:
							last_score = str(self.scorejson["data"])
							main.wsworker.send(json.dumps(self.scorejson))

			except OSError:
				self.end = True
				main.ocrstart_button.setEnabled(True)
				self.kill()

			cv2.waitKey(1)

		cap.release()
		cv2.destroyAllWindows()
		self.kill()

	def kill(self):
		self.terminate()

class create_mode_window(QWidget):
	def __init__(self, settings):
		QWidget.__init__(self)
		try:
			shutil.rmtree(_application_path + "\\temp")
		except OSError as e:
			pass

		self.createmode = -1
		self.font = settings["font"]

		self.setWindowTitle("Font Generator")
		self.setWindowIcon(main.icon)
		self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowMinMaxButtonsHint)
		self.setFixedWidth(main.small_window_width)

		self.dir_label = QLabel("Directory for font (if existing, it will add to the original)")
		self.dir_label.setWordWrap(True);

		self.dir_prompt = QLineEdit(u"" + self.font + "")
		self.start_stop = QPushButton("Start")
		self.start_stop.setFocus()

		self.start_stop.clicked.connect(self.start)

		self.main_layout = QGridLayout()
		self.main_layout.addWidget(self.dir_label, 0, 0, 1, 1)
		self.main_layout.addWidget(self.dir_prompt, 1, 0, 1, 1)
		self.main_layout.addWidget(self.start_stop, 2, 0, 1, 1)

		self.setLayout(self.main_layout)

	def get_create_mode(self):
		return (self.createmode, self.font)

	def start(self):
		self.dir_prompt.setEnabled(False)
		self.start_stop.setText("Stop")
		self.start_stop.clicked.connect(self.stop)
		self.font = self.dir_prompt.text()
		self.createmode = 0

	def stop(self):
		main.font = self.font
		self.createmode = -1
		main.rename_digits()
		self.close()

	def closeEvent(self, event):
		self.stop()

class create_mode_rename(QWidget):
	def __init__(self, settings):
		QWidget.__init__(self)
		self.loaded_images = []
		self.ref_images_inc = [0] * 10
		self.font = "create"
		self.write_font = main.font
		self.progress = 0
		self.done = False
		self.temp_font = main.temp_font

		self.digit_size = (180, 280)

		self.setWindowTitle("Assign Digits")
		self.setWindowIcon(main.icon)
		self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowMinMaxButtonsHint)
		self.setFixedWidth(main.small_window_width)

		self.instruction_label = QLabel("Press the number key corresponding to the digit, or 'n' to ignore.")
		self.instruction_label.setWordWrap(True);
		self.go_button = QPushButton("Go")
		self.go_button.clicked.connect(self.run)

		self.image_label = QLabel()
		self.image_label.setFixedSize(QSize(self.digit_size[0], self.digit_size[1]))
		self.image_label.setAlignment(Qt.AlignCenter)
		self.text_box = QLineEdit(u"")
		self.text_box.textChanged.connect(self.next)

		self.main_layout = QGridLayout()
		self.main_layout.addWidget(self.instruction_label, 0, 0, 1, 1)
		self.main_layout.addWidget(self.go_button, 1, 0, 1, 1)

		self.setLayout(self.main_layout)

	def display_webcam_feed(self, feed):
		feed = feed
		feed = cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)

		feed = cv2.resize(feed, self.digit_size, 1, 1, cv2.INTER_AREA)

		image = QImage(feed, feed.shape[1], feed.shape[0], feed.strides[0], QImage.Format_RGB888)

		image = qimage2ndarray.array2qimage(feed)

		self.image_label.setPixmap(QPixmap.fromImage(image))

	def run(self):
		self.instruction_label.setText("")
		self.main_layout.addWidget(self.image_label, 0, 0, 1, 1)
		self.main_layout.addWidget(self.text_box, 1, 0, 1, 1)

		self.setLayout(self.main_layout)

		self.ref_images_inc = main.ref_images_inc

		i = 0
		for filename in os.listdir(_application_path + "\\" + self.temp_font):
			self.loaded_images.append([cv2.imread(_application_path + "\\" + self.temp_font + "\\" + filename), filename])

		self.next()

	def next(self):
		self.text_box.setFocus()

		if (self.text_box.text().lower() == "n" or self.string_is_int(self.text_box.text()) or self.progress == 0):
			self.setWindowTitle(str(min(self.progress + 1, len(self.loaded_images))) + " of " + str(len(self.loaded_images)) + " (" + str(round((self.progress * 100) / len(self.loaded_images), 2)) + "%)")
			if not (self.text_box.text().lower() == "n" or self.progress == 0):
				print
				value = int(self.text_box.text())

				cv2.imwrite(_application_path + "\\frames\\" + self.write_font + "\\" + str(value) + "_" + str(self.ref_images_inc[value]) + ".png",
				self.loaded_images[self.progress - 1][0])

				self.ref_images_inc[value] += 1

			if self.progress < len(self.loaded_images):
				horiz = 1
				img = self.loaded_images[self.progress][0]
				if img.shape[1] == 1:
					horiz = 3
				img = cv2.copyMakeBorder(img, top=1, bottom=1, left=horiz, right=horiz, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

				self.display_webcam_feed(img)
				self.progress += 1
			else:
				self.done = True
				try:
				    shutil.rmtree(_application_path + "\\temp")
				except OSError as e:
				    pass

				self.kill()

			self.text_box.setText("")

	def kill(self):
		self.close()

	def closeEvent(self, event):
		if not self.done:
			event.ignore()
		else:
			main.reload_digits()

	def string_is_int(self, string):
		try:
			int(string)
			return True
		except:
			return False

class main_app(QWidget):
	def __init__(self):
		QWidget.__init__(self)
		self.ocrworker = None
		self.res = [1440, 720]
		self.video_size = (750, 360)
		self.section_size = [(350, 90), (350, 60), (350, 60)]
		self.digit_size = (60, 100)
		self.small_window_width = 200
		self.coords_grid_labels = ["Clock *0:00", "Clock 0*:00", "Clock 00:*0", "Clock 00:0*", "Home Score *00", "Home Score 1*0", "Home Score 10*", "Away Score *00", "Away Score 1*0", "Away Score 10*", ]
		self.temp_font = "temp"

		self.ref_images_inc = []

		settings_file_read = open('settings.json', 'r+')

		self.settings = json.loads(settings_file_read.read())

		settings_file_read.close()

		self.coords = self.settings["coords"]
		self.warp = self.settings["warp"]
		self.pinch = self.settings["pinch"]
		self.font = self.settings["font"]

		self.cam_index = self.settings["cam_index"]

		self.ref_colors = self.settings["ref_colors"]

		self.colorcount = len(self.settings["ref_colors"])

		self.draw_coords_select = -1
		self.draw_warp_select = False
		self.reload_digits_select = False
		self.show_thresh = False

		self.image_label = QLabel()
		self.image_label.setFixedSize(QSize(self.video_size[0], self.video_size[1]))

		self.showclock = QLabel()
		self.showclock.setFixedSize(QSize(self.section_size[0][0], self.section_size[0][1]))
		self.showhome = QLabel()
		self.showaway = QLabel()

		self.setWindowTitle("Sports Graphics OCR | @mjmcphe")
		self.icon_b64 = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAX0lEQVQ4je2SQQrAIAwEJ8X/fzkllEiJa1toL0L3ILjRcRPkFwb4mzG0WNwPhpn1QnqpqCmvcVGsMOVtNVIFPWpBvaxAyhsAcSgh58h1P22hXrrTkGAWWQ2aL/7B6gJ2jg0pFsJlGiwAAAAASUVORK5CYII="
		self.icon = self.icon_from_b64(self.icon_b64)
		self.setWindowIcon(self.icon)

		self.clock_label = QLabel("0:00")

		self.wsstart_button = QPushButton("WS Start")
		self.wsstart_button.clicked.connect(self.ws_start)

		self.ocrstart_button = QPushButton("OCR Start")
		self.ocrstart_button.clicked.connect(self.ocr_start)
		self.ocrstart_button.setEnabled(False)

		self.reload_digits_button = QPushButton("Reload Digits")
		self.reload_digits_button.clicked.connect(self.reload_digits)

		self.draw_warp_button = QPushButton("Update Warp")
		self.draw_warp_button.clicked.connect(self.draw_warp)

		self.thresh_check = QCheckBox("Show Thresholding?")
		self.thresh_check.stateChanged.connect(self.update_thresh_check)

		self.send_button = QPushButton("Send")
		self.send_button.clicked.connect(self.send)

		self.cam_index_input = QLineEdit(u"" + str(self.cam_index) + "")
		self.font_input = QLineEdit(u"" + str(self.font) + "")
		self.font_gen_button = QPushButton("Generate Font")
		self.font_gen_button.clicked.connect(self.create_font_prompt)

		self.ocr_live_buttons = [self.reload_digits_button, self.draw_warp_button, self.send_button, self.font_gen_button]

		self.main_layout = QGridLayout()
		self.main_layout.addWidget(self.image_label, 0, 0, 3, 5)
		self.main_layout.addWidget(self.showclock, 0, 5, 1, 2)
		self.main_layout.addWidget(self.showhome, 1, 5, 1, 2)
		self.main_layout.addWidget(self.showaway, 2, 5, 1, 2)
		self.main_layout.addWidget(self.clock_label, 4, 0, 1, 3)
		self.main_layout.addWidget(QLabel("Camera Index:"), 4, 3, 1, 1)
		self.main_layout.addWidget(QLabel("Font:"), 4, 4, 1, 1)
		self.main_layout.addWidget(self.wsstart_button, 5, 0, 1, 2)
		self.main_layout.addWidget(self.ocrstart_button, 5, 2, 1, 1)
		self.main_layout.addWidget(self.cam_index_input, 5, 3, 1, 1)
		self.main_layout.addWidget(self.font_input, 5, 4, 1, 1)
		self.main_layout.addWidget(self.font_gen_button, 5, 5, 1, 1)
		self.main_layout.addWidget(self.reload_digits_button, 5, 6, 1, 1)
		self.main_layout.addWidget(self.thresh_check, 23, 0, 1, 1)
		self.main_layout.addWidget(self.send_button, 23, 6, 1, 1)


		self.coords_grid = []

		self.coords_grid.append([QLabel(""), QLabel("Left"), QLabel("Top"), QLabel("Right"), QLabel("Bottom")])
		self.coords_grid.append([QLabel("Warp/Pinch"), QLineEdit(u"" + str(self.pinch[0]) + ""), QLineEdit(u"" + str(self.pinch[1]) + ""), QLineEdit(u"" + str(self.pinch[2]) + ""), QLineEdit(u"" + str(self.pinch[1]) + ""), self.draw_warp_button])

		self.coords_grid.append([QLabel("Hue"), QLabel("Saturation"), QLabel("Value"), QLabel("Hue Range"), QLabel("Saturation Range"), QLabel("Value Range")])
		self.coords_grid.append([QLineEdit(u"" + str(self.ref_colors["0"]["color"][0]) + ""), QLineEdit(u"" + str(self.ref_colors["0"]["color"][1]) + ""), QLineEdit(u"" + str(self.ref_colors["0"]["color"][2]) + ""), QLineEdit(u"" + str(self.ref_colors["0"]["range"][0]) + ""), QLineEdit(u"" + str(self.ref_colors["0"]["range"][1]) + ""), QLineEdit(u"" + str(self.ref_colors["0"]["range"][2]) + "")])
		self.coords_grid.append([QLineEdit(u"" + str(self.ref_colors["1"]["color"][0]) + ""), QLineEdit(u"" + str(self.ref_colors["1"]["color"][1]) + ""), QLineEdit(u"" + str(self.ref_colors["1"]["color"][2]) + ""), QLineEdit(u"" + str(self.ref_colors["1"]["range"][0]) + ""), QLineEdit(u"" + str(self.ref_colors["1"]["range"][1]) + ""), QLineEdit(u"" + str(self.ref_colors["1"]["range"][2]) + "")])

		self.coords_grid.append([QLabel(""), QLabel("X1"), QLabel("X2"), QLabel("Y1"), QLabel("Y2")])

		for i in range(10):
			self.coords_grid.append([QLabel(self.coords_grid_labels[i]), QLineEdit(u"" + str(self.coords[str(i)]["x1"]) + ""), QLineEdit(u"" + str(self.coords[str(i)]["x2"]) + ""), QLineEdit(u"" + str(self.coords[str(i)]["y1"]) + ""), QLineEdit(u"" + str(self.coords[str(i)]["y2"]) + ""), QPushButton("Draw Digit")])
			self.ocr_live_buttons.append(self.coords_grid[i + 6][5])

			self.coords_grid[i + 6][5].clicked.connect((lambda j: (lambda: self.draw_coords(j)))(i))

		for button in self.ocr_live_buttons:
			button.setEnabled(False)

		for index, row in enumerate(self.coords_grid):
			for inner_index, item in enumerate(row):
				self.main_layout.addWidget(item, index + 7, (0 if inner_index == 0 else inner_index + 1), 1, (2 if inner_index == 0 else 1))

		self.setLayout(self.main_layout)
		self.move(300,50)

	def icon_from_b64(self, base64):
	    pixmap = QtGui.QPixmap()
	    pixmap.loadFromData(QtCore.QByteArray.fromBase64(base64))
	    icon = QtGui.QIcon(pixmap)
	    return icon

	def draw_coords(self, coord):
		self.draw_coords_select = coord

	def reload_digits(self):
		self.send()
		self.reload_digits_select = True

	def draw_warp(self):
		self.draw_warp_select = True

	def update_thresh_check(self):
		self.show_thresh = self.thresh_check.isChecked()

	def create_font_prompt(self):
		self.create_window = create_mode_window(self.settings)
		self.create_window.show()

	def rename_digits(self):
		self.rename_window = create_mode_rename(self.settings)
		self.rename_window.show()

	def ws_start(self):
		self.setWindowTitle("Sports Graphics OCR | @mjmcphe (" + self.get_ip() + ")")
		self.wsworker = WebSocketsWorker()
		self.wsworker.start()

		self.wsstart_button.setEnabled(False)
		self.ocrstart_button.setEnabled(True)

	def get_ip(self):
	    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	    try:
	        s.connect(('10.255.255.255', 1))
	        ip = s.getsockname()[0]
	    except Exception:
	        ip = '127.0.0.1'
	    finally:
	        s.close()
	    return ip

	def ocr_start(self):
		try:
			self.draw_coords_select = -1
			self.cam_index = int(self.cam_index_input.text())
			self.font = self.font_input.text()
			self.send()

			testcap = cv2.VideoCapture(self.cam_index)
			if testcap is None or not testcap.isOpened():
				print("Camera doesn't Exist")
			else:
				self.cam_index_input.setEnabled(False)
				for button in self.ocr_live_buttons:
					button.setEnabled(True)

				self.ocrstart_button.setText("OCR Restart")
				self.ocrstart_button.clicked.connect(self.ocr_stop)
				self.ocrworker = ocr_worker(self.settings)
				self.ocrworker.start()
				self.send();
		except AttributeError:
			print("Invalid Camera")
			self.ocr_stop()
			self.coords_grid[1][0].setEnabled(True)
			self.coords_grid[1][1].setEnabled(True)


	def ocr_stop(self):
		self.ocrworker.kill()

	def close_all(self):
		self.close()

	def update_text_boxes(self):
		for i in range(10):
			self.coords_grid[i + 6][1].setText(str(self.coords[str(i)]["x1"]))
			self.coords_grid[i + 6][2].setText(str(self.coords[str(i)]["x2"]))
			self.coords_grid[i + 6][3].setText(str(self.coords[str(i)]["y1"]))
			self.coords_grid[i + 6][4].setText(str(self.coords[str(i)]["y2"]))
		self.send()

	def keyPressEvent(self, e):
		if e.key() == QtCore.Qt.Key_Return or e.key() == QtCore.Qt.Key_Enter:
			self.send()

	def send(self):
		oldcoords = self.coords
		oldwarp = self.warp
		oldcolors = self.ref_colors
		oldpinch = self.pinch
		oldfont = self.font

		for i in range(10):

			self.coords[str(i)]["x1"] = int(self.coords_grid[i + 6][1].text())
			self.coords[str(i)]["x2"] = int(self.coords_grid[i + 6][2].text())
			self.coords[str(i)]["y1"] = int(self.coords_grid[i + 6][3].text())
			self.coords[str(i)]["y2"] = int(self.coords_grid[i + 6][4].text())

			for h, j in enumerate(["x", "y"]):
				if self.coords[str(i)][str(j) + "1"] == self.coords[str(i)][str(j) + "2"]:
					if self.coords[str(i)][str(j) + "1"] > 1:
						self.coords[str(i)][str(j) + "1"] = (int(self.coords[str(i)][str(j) + "1"]) - 1)
					else:
						self.coords[str(i)][str(j) + "2"] = (int(self.coords[str(i)][str(j) + "2"]) + 1)
						self.coords_grid[i + 6][(h + 1) * 2 - 1].setText(str(int(self.coords[str(i)][str(j) + "2"]) + 1))
					self.update_text_boxes()

		self.settings["coords"] = self.coords
		#except:
			#self.coords = oldcoords

		try:
			self.settings["warp"] = self.warp
		except:
			self.warp = oldwarp

		try:
			for i in range(4):
				self.pinch[i] = int(self.coords_grid[1][i + 1].text())

			self.settings["pinch"] = self.pinch
		except:
			self.pinch = pinch

		try:
			for i in range(self.colorcount):
				for j in range(3):
					self.ref_colors[str(i)]["color"][j] = int(self.coords_grid[i + 3][j].text())
					self.ref_colors[str(i)]["range"][j] = int(self.coords_grid[i + 3][j + 3].text())

			self.settings["ref_colors"] = self.ref_colors
		except:
			self.ref_colors = oldcolors

		try:
			self.font = self.font_input.text()
			self.settings["font"] = self.font
		except:
			self.font = oldfont

		settings_file_write = open('settings.json', 'w')
		settings_file_write.write(json.dumps(self.settings))
		settings_file_write.close()

	def check_coords(self):
		return self.coords

	def check_ref_color(self):
		return self.ref_colors

	def check_pinch(self):
		return self.pinch

	def set_clock_text(self, text):
		self.clock_label.setText(text)

	def display_webcam_feed(self, feed, index):
		feed = feed
		feed = cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)

		display = None

		if index == -1:
			feed = cv2.resize(feed, self.video_size, 1, 1, cv2.INTER_NEAREST)
			image = QImage(feed, feed.shape[1], feed.shape[0], feed.strides[0], QImage.Format_RGB888)
			image = qimage2ndarray.array2qimage(feed)
			self.image_label.setPixmap(QPixmap.fromImage(image))
		elif index == 0:
			feed = cv2.resize(feed, self.section_size[index], 1, 1, cv2.INTER_NEAREST)
			image = QImage(feed, feed.shape[1], feed.shape[0], feed.strides[0], QImage.Format_RGB888)
			image = qimage2ndarray.array2qimage(feed)
			self.showclock.setPixmap(QPixmap.fromImage(image))
		elif index == 1:
			feed = cv2.resize(feed, self.section_size[index], 1, 1, cv2.INTER_NEAREST)
			image = QImage(feed, feed.shape[1], feed.shape[0], feed.strides[0], QImage.Format_RGB888)
			image = qimage2ndarray.array2qimage(feed)
			self.showhome.setPixmap(QPixmap.fromImage(image))
		elif index == 2:
			feed = cv2.resize(feed, self.section_size[index], 1, 1, cv2.INTER_NEAREST)
			image = QImage(feed, feed.shape[1], feed.shape[0], feed.strides[0], QImage.Format_RGB888)
			image = qimage2ndarray.array2qimage(feed)
			self.showaway.setPixmap(QPixmap.fromImage(image))

	def closeEvent(self, event):
		try:
			shutil.rmtree(_application_path + "\\temp")
		except OSError as e:
			pass

		event.accept()
		sys.exit()

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	main = main_app()
	main.show()

	sys.exit(app.exec_())
