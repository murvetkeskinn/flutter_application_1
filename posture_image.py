import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter

from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant

def relu(x): return Activation('relu')(x)

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x, weight_decay):
    # Block 1
    x = conv(x, 64, 3, "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")

    # Block 4
    x = conv(x, 512, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)

    return x


def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x


def apply_mask(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w

def get_testing_model():
    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model


tic=0
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

model1 = get_testing_model()
model1.load_weights('./model/keras/model.h5')

def process (input_image, params, model_params):
	''' Start of finding the Key points of full body using Open Pose.'''
	oriImg = cv2.imread(input_image)  # B,G,R order
	multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
	heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
	paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
   
	for m in range(1):
		scale = multiplier[m]
		imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
		input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
		output_blobs = model1.predict(input_img)
		heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
		heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
		heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
		heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
		paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
		paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
		paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
		paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
		heatmap_avg = heatmap_avg + heatmap / len(multiplier)
		paf_avg = paf_avg + paf / len(multiplier)

	all_peaks = [] #To store all the key points which a re detected.
	peak_counter = 0
	
	prinfTick(1) #prints time required till now.

	for part in range(18):
	    map_ori = heatmap_avg[:, :, part]
	    map = gaussian_filter(map_ori, sigma=3)

	    map_left = np.zeros(map.shape)
	    map_left[1:, :] = map[:-1, :]
	    map_right = np.zeros(map.shape)
	    map_right[:-1, :] = map[1:, :]
	    map_up = np.zeros(map.shape)
	    map_up[:, 1:] = map[:, :-1]
	    map_down = np.zeros(map.shape)
	    map_down[:, :-1] = map[:, 1:]

	    peaks_binary = np.logical_and.reduce(
	        (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
	    peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
	    peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
	    id = range(peak_counter, peak_counter + len(peaks))
	    peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

	    all_peaks.append(peaks_with_score_and_id)
	    peak_counter += len(peaks)

	connection_all = []
	special_k = []
	mid_num = 10

	prinfTick(2) #prints time required till now.
	print()
	position ,degrees = checkPosition(all_peaks) #check position of spine.
	checkKneeling(all_peaks) #check whether kneeling oernot
	checkHandFold(all_peaks) #check whether hands are folding or not.
	canvas1 = draw(input_image,all_peaks) #show the image.
	return canvas1 , position,degrees


def draw(input_image, all_peaks):
    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    return canvas


def checkPosition(all_peaks):
    try:
        f = 0
        if (all_peaks[16]):
            a = all_peaks[16][0][0:2] #Right Ear
            f = 1
        else:
            a = all_peaks[17][0][0:2] #Left Ear
        b = all_peaks[11][0][0:2] # Hip
        angle = calcAngle(a,b)
        degrees = round(math.degrees(angle))
        if (f):
            degrees = 180 - degrees
        if (degrees<70):
            return 1,degrees
        elif (degrees > 110):
            return -1,degrees
        else:
            return 0,degrees
    except Exception as e:
        print("person not in lateral view and unable to detect ears or hip")

#calculate angle between two points with respect to x-axis (horizontal axis)
def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if (ax == bx):
            return 1.570796
        return math.atan2(by-ay, bx-ax)
    except Exception as e:
        print("unable to calculate angle")


def checkHandFold(all_peaks):
	try:
		if (all_peaks[3][0][0:2]):
			try:
				if (all_peaks[4][0][0:2]):
					distance  = calcDistance(all_peaks[3][0][0:2],all_peaks[4][0][0:2]) #distance between right arm-joint and right palm.
					armdist = calcDistance(all_peaks[2][0][0:2], all_peaks[3][0][0:2]) #distance between left arm-joint and left palm.
					if (distance < (armdist + 100) and distance > (armdist - 100) ): #this value 100 is arbitary. this shall be replaced with a calculation which can adjust to different sizes of people.
						print("Not Folding Hands")
					else: 
						print("Folding Hands")
			except Exception as e:
				print("Folding Hands")
	except Exception as e:
		try:
			if(all_peaks[7][0][0:2]):
				distance  = calcDistance( all_peaks[6][0][0:2] ,all_peaks[7][0][0:2])
				armdist = calcDistance(all_peaks[6][0][0:2], all_peaks[5][0][0:2])
				# print(distance)
				if (distance < (armdist + 100) and distance > (armdist - 100)):
					print("Not Folding Hands")
				else: 
					print("Folding Hands")
		except Exception as e:
			print("Unable to detect arm joints")
		

def calcDistance(a,b): #calculate distance between two points.
	try:
		x1, y1 = a
		x2, y2 = b
		return math.hypot(x2 - x1, y2 - y1)
	except Exception as e:
		print("unable to calculate distance")

def checkKneeling(all_peaks):
	f = 0
	if (all_peaks[16]):
		f = 1
	try:
		if(all_peaks[10][0][0:2] and all_peaks[13][0][0:2]): # if both legs are detected
			rightankle = all_peaks[10][0][0:2]
			leftankle = all_peaks[13][0][0:2]
			hip = all_peaks[11][0][0:2]
			leftangle = calcAngle(hip,leftankle)
			leftdegrees = round(math.degrees(leftangle))
			rightangle = calcAngle(hip,rightankle)
			rightdegrees = round(math.degrees(rightangle))
		if (f == 0):
			leftdegrees = 180 - leftdegrees
			rightdegrees = 180 - rightdegrees
		if (leftdegrees > 60  and rightdegrees > 60): # 60 degrees is trail and error value here. We can tweak this accordingly and results will vary.
			print ("Both Legs are in Kneeling")
		elif (rightdegrees > 60):
			print ("Right leg is kneeling")
		elif (leftdegrees > 60):
			print ("Left leg is kneeling")
		else:
			print ("Not kneeling")

	except IndexError as e:
		try:
			if (f):
				a = all_peaks[10][0][0:2] # if only one leg (right leg) is detected
			else:
				a = all_peaks[13][0][0:2] # if only one leg (left leg) is detected
			b = all_peaks[11][0][0:2] #location of hip
			angle = calcAngle(b,a)
			degrees = round(math.degrees(angle))
			if (f == 0):
				degrees = 180 - degrees
			if (degrees > 60):
				print ("Both Legs Kneeling")
			else:
				print("Not Kneeling")
		except Exception as e:
			print("legs not detected")



def showimage(img): #sometimes opencv will oversize the image when using using `cv2.imshow()`. This function solves that issue.
    screen_res = 1280, 720 #my screen resolution.
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prinfTick(i): #Time calculation to keep a trackm of progress
    toc = time.time()
    print ('processing time%d is %.5f' % (i,toc - tic))   

def showimage1(img): #sometimes opencv will oversize the image when using using `cv2.imshow()`. This function solves that issue.

    
	vi=False
	if(vi == False):
	    time.sleep(2)
	    params, model_params = config_reader()
	    canvas, position, degrees = process(img, params, model_params)

        
	return canvas,position, 100*abs(degrees-90)/90.0

if __name__ == '__main__': #main function of the program
	tic = time.time()
	print('start processing...')

	model = get_testing_model()
	model.load_weights('./model/keras/model.h5')

	vi=False
	if(vi == False):
	    time.sleep(2)
	    params, model_params = config_reader()
	    canvas, position,degrees= process('./sample_images/test.jpg', params, model_params)
	    dbo=100* abs(degrees-90)/90.0
        
	    if (position == 1):
	    	print("Kambur Duruş", dbo)
	    elif (position == -1):
	    	print ("Uzanmış Duruş",dbo)
	    else:
	    	print("Dik Duruş",degrees)
            
showimage(canvas)


