# -*- coding:utf-8 -*-
import cv2
import sys
import os
import shutil

args = sys.argv
argc = len(args)

#if(argc != 2):
#	print '引数を指定して実行してください。'
#	quit()

#image_path = args[1]
image_path = "c:/temp/face.jtif"

cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"

#ファイル読み込み
image = cv2.imread(image_path)
if(image is None):
	print('can not open image')
	quit()

#グレースケール変換
image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)

#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

#物体認識（顔認識）の実行
facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

print("face rectangle")
print(facerect)

#ディレクトリの作成
if len(facerect) > 0:
	path = os.path.splitext(image_path)
	dir_path = path[0] + '_face'
	if os.path.isdir(dir_path):
		shutil.rmtree(dir_path)
	os.mkdir(dir_path)

i = 0
for rect in facerect:
	#顔だけ切り出して保存
	x = rect[0]
	y = rect[1]
	width = rect[2]
	height = rect[3]
	dst = image[y:y+height, x:x+width]
	new_image_path = dir_path + '/' + str(i) + path[1]
	cv2.imwrite(new_image_path, dst)
	i += 1

if len(facerect) > 0:
	color = (255, 255, 255) #白
	for rect in facerect:
		#検出した顔を囲む矩形の作成
		cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), color, thickness=2)

	#認識結果の保存
	new_image_path = dir_path + '/' +'all' + path[1]
	cv2.imwrite(new_image_path, image)