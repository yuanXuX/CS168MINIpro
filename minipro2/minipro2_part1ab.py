import csv
from functools import reduce 
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy.spatial as sp
import datetime


articles=[]
labels=[]
groups=[]

def getData():
	currentID=1	
	#matrix=np.zeros(1000,61100)
	count=0
	with open('D:/算法分析与设计/minipro2/p2_data/data50.csv', 'rt') as csvfile:
		reader=csv.reader(csvfile)
		wordCts={}
		for row in reader:
			articleID=int(row[0])
			if articleID != currentID:
			#	count+=1
				articles.append(dict(wordCts))
				wordCts={}
				currentID=articleID
			wordID=int(row[1])
			wordCount=row[2]
			wordCts[wordID]=wordCount
			#matrix[count,wordID]=row[2]
		articles.append(dict(wordCts))

def getLabel():
	currentID=1
	with open('D:/算法分析与设计/minipro2/p2_data/label.csv', 'rt') as csvfile:
		reader=csv.reader(csvfile)   
		articleInclude=[]
		i=0
		for row in reader:
			groupID=int(row[0])
			if groupID!=currentID:
				labels.append(list(articleInclude))
				articleInclude=[]
				currentID=groupID
			articleInclude.append(i)
			i+=1
			
		labels.append(list(articleInclude))

def getGroup():
	with open('D:/算法分析与设计/minipro2/p2_data/groups.csv', 'rt') as csvfile:
		reader=csv.reader(csvfile)
		for row in reader:
			groups.append(row[0])


'''
def jaccard(x,y,words):
	return np.sum(np.minimum(x,y))/np.sum(np.maximum(x,y))
	

def l2(x,y):
	return -np.sqrt(np.sum((x-y)**2))

def cosine(x,y):
	return np.sum(x*y)/(np.sum(np.abs(x))*np.sum(np.abs(y)))

def getSimilarity(group1,group2,words,method):
	total_similarity=0.0
	num=0.0
	for article1 in group1:
		for article2 in group2:
			similarity=method(words[article1],words[article2])
			num+=1.0
			total_similarity+=similarity
	return total_similarity/num
'''
def jaccard(x,y,words):
	nu=0
	de=0
	for i in words:
		#找到word_i,没有就返回0
		xi=x.get(i,0)
		yi=y.get(i,0)
		nu+=min(int(xi),int(yi))
		de+=max(int(xi),int(yi))
	return float(nu)/de

def l2(x,y,words):
	simi=0
	for i in words:
		xi=x.get(i,0)
		yi=y.get(i,0)
		simi+=math.pow(int(xi)-int(yi),2)
	simi=math.sqrt(simi)
	return -1*simi

def cosine(x,y,words):
	nu=0
	#xx yy 范数
	xx=0
	yy=0
	for i in words:
		xi=x.get(i,0)
		yi=y.get(i,0)
		nu+=int(xi)*int(yi)
		xx+=math.pow(int(xi),2)
		yy+=math.pow(int(yi),2)
	xx=math.sqrt(xx)
	yy=math.sqrt(yy)
	return float(nu)/(xx*yy)

def getSimi(method,filename):
	groupsNum=len(groups)
	matrix=np.zeros((groupsNum,groupsNum))
	for i in range(groupsNum):
		for j in range(groupsNum):
			groupA=labels[i]
			groupB=labels[j]
			totalNum=0
			simiSum=0
			for article1 in groupA:
				for article2 in groupB:
					x=articles[article1]
					y=articles[article2]
					#字典的并集变成集合
					#wordsUnion=dict(x,**y)
					wordList = reduce(set.union, map(set, map(dict.keys, [x, y])))
					simiSum+=method(x,y,wordList)
					totalNum+=1
			matrix[i][j]=float(simiSum)/totalNum
	makeHeatMap(matrix,groups,plt.cm.Blues,filename)
			
#参考CS168 minipro2部分
def makeHeatMap(data, names, color, outputFileName):
    #to catch "falling back to Agg" warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #code source: http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
        fig, ax = plt.subplots()
        #create the map w/ color bar legend
        heatmap = ax.pcolor(data, cmap=color)
        cbar = plt.colorbar(heatmap)
        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.set_xticklabels(range(1,21))
        ax.set_yticklabels(names)
        plt.tight_layout()
        plt.savefig(outputFileName, format = 'png')
        plt.close()




getData()
getLabel()
getGroup()
'''
getSimi(jaccard,'jaccard.jpg')
getSimi(l2,'l2.jpg')
getSimi(cosine,'cosine.jpg')
'''
'''
starttime = datetime.datetime.now()
endtime = datetime.datetime.now()
print("优化前：")
print (endtime - starttime).seconds
'''

'''
size:int or tuple of ints
输出的shape，默认为None，只输出一个值
'''
'''
def COS(x, y):
    numerator = 0
    x_term = 0
    y_term = 0
    for i in range(len(x)):
        x_count = x[i]
        y_count = y[i]
        numerator += x_count * y_count
        x_term += math.pow(x_count, 2)
        y_term += math.pow(y_count, 2)
    x_term = math.sqrt(x_term)
    y_term = math.sqrt(y_term)
    return float(numerator)/(x_term * y_term)
'''

'''
def Reduce(d):
	Guaasin=np.random.normal(size=(d,61070))
	ReduceMatrix=[]
	for article in articles:
		row=np.zeros(61070,)
		for key,value in article.items():
			row[key-1]=value
		ReduceMatrix.append(np.dot(Guaasin,row))
	return ReduceMatrix

def FindSimi(ReduceMatrix):
	Baseline=np.zeros((20,20))
	for i in range(len(articles)):
		currentSimi=float("-inf")
		mostSimiGroupId=0
		xi=ReduceMatrix[i]
		for j in range(len(articles)):
			if i==j:
				continue
			yj=ReduceMatrix[j]
			simi=COS(xi,yj)
			if simi>currentSimi:
				currentSimi=simi
				mostSimiGroupId=j
			Baseline[int(i/50)][int(mostSimiGroupId/50)]+=1
	return Baseline

'''
maxID=60170
ReduceMatrix=Reduce(10)
newMatrix=FindSimi(ReduceMatrix)
filename="part2.jpg"
makeHeatMap(newMatrix, groups, plt.cm.Blues, filename)

#生成128个高斯矩阵
def constructMatrix(d):
	dkMatrix=[]
	for i in range(128):
		matrix=np.random.normal(size=(d,maxID))
		dkMatrix.append(matrix)
	return dkMatrix

def hash(article,dkMatrix):
	v=np.zeros(maxID)
	hashArticle=[]
	for key,value in article.items():
		v[k-1]=value
	for i in range(128):
		hashvalue=np.dot(dkMatrix[i],v)
		hashindex=np.zeros(len(hashvalue),)
		for j in range(len(hashvalue)):
			if hashvalue[j]>0:
				hashindex[j]=1
		hashArticle.append(hashindex)
	return hashArticle

def hyperplaneHash(dkMatrix):
	hashArticles=[]
	for article in articles:
		hashArticles.append(hash(article,dkMatrix))
	return hashArticles

def equ(l1,l2):
	for i in range(len(l1)):
		if l1[i]!=l2[i]:
			return False
	return True

def classfication(q,dkMatrix,hashArticles):
	hashvalue=hash(q,dkMatrix)
	simi=float("-inf")
	simiGroup=0
	sq_size=0
	for i in range(len(articles)):
		groupID=i/50
		article=articles[i]
		datapoint=hashArticles[i]

		for j in range(128):
			if equ(hashvalue[j],datapoint[j]):
				words= dict(dict1,**dict2)
				word_ids=words.keys()
				similarity=cosine(q,article,word_ids)
				if similarity>similarity:
					simi=similarity
					simiGroupID=groupID
				sq_size+=1
				break
	return (simiGroupID,sq_size)

def lsh(d):
	dkMatrix=constructMatrix(d)
	hashArticles=hyperplaneHash(dkMatrix)
	error=0
	total_sq_size=0
	for i in range(len(articles)):
		groupID=i/50
		suggested_group,sq_size=classfication(articles[i],dkMatrix,hashArticles)
		total_sq_size+=sq_size
		if groupID!=suggested_group:
			error+=1
	return(float(error)/len(articles),float(total_sq_size)/len(articles))

def fig(error,aveSQsize,outputFile):
	plt.title("error and sqSize")
	plt.axis([0,1000,0.5,1])
	plt.plot(aveSQsize,error,'ro')
	plt.savefig(outputFile,format='png')
	plt.close()

error=[]
aveSQsize=[]
for d in range(5,21):
	print("LSH for d value "+str(d))
	classerror,sqsize=lsh(d)
	error.append(classerror)
	aveSQsize.append(sqsize)
fig(error,aveSQsize,"lsh.png")












