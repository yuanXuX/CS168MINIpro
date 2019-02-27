
articles=[]
labels=[]
groups=[]
def getData():
	currentID=1
	with open('D:/算法分析与设计/minipro2/p2_data/data50.csv', 'rt') as csvfile:
		reader=csv.reader(csvfile)
		wordCts={}
		for row in reader:
			articleID=int(row[0])
			if articleID != currentID:
				articles.append(dict(wordCts))
				wordCts={}
				currentID=articleID
			wordID=int(row[1])
			wordCount=row[2]
			wordCts[wordID]=wordCount
		articles.append(dict(wordCts))

def getLabel():
	currentID=1
	with open('D:/算法分析与设计/minipro2/p2_data/data50.csv', 'rt') as csvfile:
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
	with open('D:/算法分析与设计/minipro2/p2_data/data50.csv', 'rt') as csvfile:
		reader=csv.reader(csvfile)
		for row in reader:
			groups.append(row[0])

