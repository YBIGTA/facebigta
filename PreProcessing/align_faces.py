from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os

## parse command-line arguments
ap = argparse.ArgumentParser()

## argument input with -r or --root_dir: root folder of the original image sources
ap.add_argument("-r", "--root_dir", required=True,
help="path to root directories of input images")

## argument input with -d or --des_dir: path to destination directories of output images
ap.add_argument("-d", "--des_dir", required=True,
help="path to destination directories of output images")
args = vars(ap.parse_args())


## load dlib's face detector
detector = dlib.get_frontal_face_detector()

## load dlib's shape predictor using 68 dots to extract facial features such as eyes, noes, chins
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=128)

root_dir = args["root_dir"]
des_dir = args["des_dir"].split("/")[-1]

## if directory does not exits, make it
if not os.path.exists(des_dir):
	os.mkdir(des_dir)

## recursively explore sub-directories of the root and get sources
input_files = [os.path.join(dp,f) for dp, dn, fn in os.walk(os.path.expanduser(root_dir)) for f in fn]

# loop over the face detections
for input_file in input_files:

	## load image
	image = cv2.imread(input_file)
	image = imutils.resize(image, width=800)

	## conver to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	des_path = '/'.join(['..',des_dir] + input_file.split("/")[2:-1])
	des_list = des_path.split("/")

	## make sub-directories in des_dir, if they don't exist
	for i in range(len(des_list)):
		elem = des_list[:i+1]
		if elem and not os.path.exists("/".join(elem)):
			os.mkdir("/".join(elem))

	file_name = input_file.split("/")[-1]
	out_file = os.path.join(des_path,file_name)

	# show the original input image and detect faces in the grayscale
	# image
	rects = detector(gray, 2)
	for rect in rects:
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks
		try:
			(x, y, w, h) = rect_to_bb(rect)
			faceOrig = imutils.resize(image[y:y + h, x:x + w], width=128)
			faceAligned = fa.align(image, gray, rect)

			# display the output images
			cv2.imwrite(out_file,faceAligned)
			cv2.waitKey(0)
		except:
			continue
