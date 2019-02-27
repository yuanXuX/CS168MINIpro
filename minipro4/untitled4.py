# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:56:49 2018

@author: xuyuan
"""

from numpy.random import randn



# 2CCCCCCCC

cs = [c * .05 for c in range(11)]

# print(cs)



def make_Y(c):

	Y = np.array([2 * i * .001 for i in range(1, 1001)])

	noise = randn(1000) * math.sqrt(c)

	Y += noise

	return Y



import time 



def make_X(c):

	X = np.array([i * .001 for i in range(1, 1001)])

	test_rand = randn(1000)

	test_c = math.sqrt(c)

	noise = randn(1000) * math.sqrt(c)

	X += noise

	#print("shape of X: ", X.shape)

	return X





def make_plot_2(filename, title, X, flag):

	# c on horizontal axis

	# pca-recover on vertical - red dot

	# ls-recover on vertical - blue dot

	plt.title(title)

	plt.xlabel("Noise Level")

	plt.ylabel("Slope")

	

	for i in range(30):

		for c in cs:

			# noise = randn(1000) * np.sqrt(c)

			Y = make_Y(c)

			if flag != 'c':

				X = make_X(c)

			# print(X)

			# print("X", np.sum(X))

			# print("Y", np.sum(Y))



			# if (np.sum(Y) / np.sum(X)) >= 2:

			# 	print(c)

			# 	print("X", np.sum(X))

			# 	print("Y", np.sum(Y))

			pca = pca_recover(X, Y)

			print("value of pca: ", pca)

			ls = ls_recover(X, Y)

			plt.plot(c, pca, 'rs', label = 'pca', alpha=0.3)

			plt.plot(c, ls, 'bs', label = 'ls', alpha=0.3)

			#X = None

	plt.savefig(filename + ".png", format = 'png')

	plt.close()



X = [x * .001 for x in range(1, 1001)]

# make_plot_2("2c", "Noise on Y", X, flag = 'c')

# make_plot_2("2d", "Noise on X and Y", X, flag = 'd')



# Y = np.array([2 * i * .001 for i in range(1, 1001)])

# noise = randn(1000) * math.sqrt(0.4)

# noisy_Y = Y + noise



# noise = randn(1000) * math.sqrt(0.4)

# noisy_X = X + noise



# def make_plot_temp(filename, x, y):	

# 	plt.scatter(x, y)

# 	plt.savefig(filename + ".png", format = 'png')

# 	plt.close()



# make_plot_temp("noisy on y and x, c = 0.4", noisy_Y, noisy_X)



	





# identifiers, sexes, population_tag, nucleobases = process_text_file('p4dataset2018.txt')

# nucleobases_binary = convert_array_to_binary(nucleobases)