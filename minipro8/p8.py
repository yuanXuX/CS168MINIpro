# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:21:56 2018

@author: xuyuan
"""
import scipy.io.wavfile as wavfile 
import numpy as np
import matplotlib.pyplot as plt



# question 2
def pro_x_y(x, y):
	lengthOfx = len(x)
	lengthOfy = len(y)
	xr = x
	yr = y
	if lengthOfx < lengthOfy:
		xr += [0 for i in range(lengthOfy - lengthOfx)]
	elif lengthOfy < lengthOfx:
		yr += [0 for i in range(lengthOfx - lengthOfy)]
	xr += [0 for i in range(len(xr))]
	yr += [0 for i in range(len(yr))]
	return xr, yr

def multiply(x, y):
	xr, yr = pro_x_y(x, y)
	x_fft = np.fft.fft(xr)
	y_fft = np.fft.fft(yr)
	x_y_mult = np.multiply(x_fft, y_fft)
	inv = np.fft.ifft(x_y_mult)
	values = []
	carry_over = 0
	for val in inv:
		print(val)
		curr = int(round(val.real, 0) + carry_over)
		if curr >= 10:
			carry_over = int(curr)//10
			curr %= 10
		else:
			carry_over = 0
		values.append(curr)
	while(values[-1] == 0):
		del values[-1]
	return values	
x = [0,9,8,7,6,5,4,3,2,1,0,9,8,7,6,5,4,3,2,1]
y = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
#print(multiply(x, y))






#part3
#b
file = 'laurel_yanny.wav'
sampleRate, data = wavfile.read(file)
print("sample rate: ", sampleRate)
print("shape of data: ", data.shape)

def plot_b(data):
	time = data.shape[0]
	x= [i for i in range(time)]
	plt.plot(x, data)
	plt.title("part3_b")
	plt.xlabel("Time")
	plt.ylabel("Phsyical Position")
	plt.savefig('p3_b.png', format = 'png')
	plt.close()
#plot_b(data)

#c

def part3_c(data):
    data_fft = np.fft.fft(data)
    print("shape of transformed data: ", data_fft.shape)
    print(data_fft[0])
    x = [i for i in range(data_fft.shape[0])]
    plt.plot(x, np.absolute(data_fft))
    plt.title("FFT")
    plt.xlabel("Time")
    plt.ylabel("Fourier Transform Magnitude")
    plt.savefig("p3_c.png", fomrat = 'png')
    plt.close()
#part3_c(data)

#d
#part d



def part3_d(data):
	blocks = 500
	max_feq = 80
	total_chunks = data.shape[0] // blocks
	fourier_matrix = np.zeros((total_chunks, max_feq))
	for i in range(total_chunks):
		current_chunk = data[i * blocks : (i + 1) * blocks]
		curr_fourier = np.fft.fft(current_chunk)
		curr_fourier = np.absolute(curr_fourier)
		fourier_matrix[i, ] = curr_fourier[:80]
	#fourier_matrix = np.absolute(fourier_matrix)
	fourier_matrix = np.sqrt(fourier_matrix)
	plt.imshow(fourier_matrix, cmap = 'hot')
	plt.xlabel("Chunk Index")
	plt.ylabel("Fourier Coefficients")
	plt.title("paert3_d")
	plt.savefig('p3_d.png', format = 'png')
	plt.close()
#part3_d(data)


#e
def part3_e1(data, threshold, low = False):
	data = (np.absolute(data) * 1.0 / np.max(np.absolute(data)) * 42000).astype(np.int16)
	with open(str(threshold) + ".wav", "wb") as f:
		sample_rate = sampleRate
		if low:
			sample_rate *= 1.3
		sample_rate = int(sample_rate)
		wavfile.write(f, sample_rate, data)
def part3_e2(data):
    thresholds = [40000]
    transformed_data = np.fft.fft(data)
    for threshold in thresholds:
    	high = transformed_data.copy()
    	high[:threshold] = 0
    	high[:43008 - threshold] = 0
    	low = transformed_data.copy()
    	low[threshold:] = 0
    	low[:43008 - threshold] = 0
    	high_fft = np.fft.ifft(high)
    	low_fft = np.fft.ifft(low)
    	part3_e1(high_fft, "bigger" + str(threshold))
    	part3_e1(low_fft, "smaller" + str(threshold), low = True)
part3_e2(data)

















