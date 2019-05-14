import csv
import sys
from random import shuffle
import math
from operator import itemgetter

#Global values and parameters
dataSet=[]
learningRate=.02
MAX_ITERATIONS=100
totalAttr=0
totalFields=0
totalSets=0
no_TrainSets=0
no_TestSets=0
setSize=0
k=2


#Parse dataset
def parseCSV(filename):
	fields = []
	rows = []
	global totalAttr,totalFields,totalSets,no_TrainSets,no_TestSets,setSize

	with open(filename, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		fields = next(csvreader)
		nameFlag=1
		className=""
		for row in csvreader:
			if(nameFlag):
				className=row[totalFields-1]
				nameFlag=0
			if row[totalFields-1]==className:
				row[totalFields-1]=1
			else:
				row[totalFields-1]=0
			rows.append(row)
	for row in rows:
		for i in range(len(row)):
			if(i!=(len(row)-1)):
				row[i]=float(row[i])

	totalFields=len(fields)
	totalAttr=totalFields-1
	totalSets=len(rows)
	setSize=(totalSets)//10
	no_TrainSets=9*setSize
	no_TestSets=totalSets-no_TrainSets
	return fields,rows

def calEuclidDistance(train_st,test_i):
	train=train_st
	trainNo=0
	distArr=[]
	while(trainNo<no_TrainSets):
		dist=0
		for attr in range(totalAttr):
			dist+=( (dataSet[test_i][attr]-dataSet[train][attr])*(dataSet[test_i][attr]-dataSet[train][attr]) )
		distArr.append([dist,train])
		train=(train+1)%totalSets
		trainNo+=1
	return distArr

def predict(distArr):
	distArr=sorted(distArr,key=itemgetter(0))
	topKClassCount=[0,0]
	for i in range(k):
		topKClassCount[dataSet[distArr[i][1]][-1]]+=1
	if topKClassCount[0] > topKClassCount[1]:
		return 0
	else:
		return 1

def testModel(train_st,test_st):
	falsePos=truePos=falseNeg=trueNeg=posCount=negCount=0
	testNo=0
	test=test_st
	while testNo<no_TrainSets:
		distArr=calEuclidDistance(train_st,test)
		predOp=predict(distArr)
		correctOp=dataSet[test][totalFields-1]
		if(correctOp==0):
			negCount+=1
		else:
			posCount+=1
		if(predOp==correctOp):
			if(predOp==1):
				truePos+=1
			else:
				trueNeg+=1
		else:
			if(predOp==1):
				falsePos+=1
			else:
				falseNeg+=1
		testNo+=1
		test=(test+1)%totalSets
	return truePos,falsePos, trueNeg, falseNeg

#10 fold cross validation
def trainAndCrossValidate():
	kFold=0
	train_st=0	
	net_precisionPos=net_precisionNeg=net_recallPos=net_recallNeg=0
	net_precisionPosCount=net_precisionNegCount=net_recallPosCount=net_recallNegCount=0
	net_accuracy=net_error=0

	while(kFold<10):

		#print("Fold No.",kFold+1)
		test_st=(train_st+no_TrainSets)%totalSets
		truePos,falsePos,trueNeg,falseNeg=testModel(train_st,test_st)

		if((truePos+falsePos)!=0):
			precisionPos=(truePos)/(truePos+falsePos)
			net_precisionPos+=(precisionPos)
			net_precisionPosCount+=1
			#print("Precision_Pos %=",precisionPos*100,end='\t')
		if((truePos+falseNeg)!=0):
			recallPos=(truePos)/(truePos+falseNeg)
			net_recallPos+=(recallPos)
			net_recallPosCount+=1
			#print("Recall_Pos %=",recallPos*100)
		if((trueNeg+falseNeg)!=0):
			precisionNeg=(trueNeg)/(trueNeg+falseNeg)
			net_precisionNeg+=(precisionNeg)
			net_precisionNegCount+=1
			#print("Precision_Neg %=",precisionNeg*100,end='\t')
		if((trueNeg+falsePos)!=0):
			recallNeg=(trueNeg)/(trueNeg+falsePos)
			net_recallNeg+=(recallNeg)
			net_recallNegCount+=1
			#print("Recall_Neg %=",recallNeg*100)
		accuracy=((trueNeg+truePos)/(trueNeg+truePos+falseNeg+falsePos))
		net_accuracy+=accuracy
		#print("Accuracy %=",accuracy*100)
		train_st=(train_st+setSize)%totalSets
		kFold+=1
		#print("***************************************************")

	net_precisionPos/=net_precisionPosCount
	net_recallPos/=net_recallPosCount
	net_precisionNeg/=net_recallNegCount
	net_recallNeg/=net_recallNegCount
	net_accuracy/=10
	# print("Average Recall_Pos %=",net_recallPos*100,"\nAverage Precision_Pos %=",net_precisionPos*100)
	# print("Average Recall_Neg %=",net_recallNeg*100,"\nAverage Precision_Neg %=",net_precisionNeg*100)
	# print("Accuracy %=",net_accuracy*100)
	return net_accuracy*100

def main():
	global dataSet,totalAttr,totalFields,k
	filename=sys.argv[1]
	fields, dataSet=parseCSV(filename)
	shuffle(dataSet)
	accArr=[]
	for _ in range(30):
		print("***********")
		k+=1
		print("k=",k)
		acc=trainAndCrossValidate()
		accArr.append([k,acc])
		print("Net Accuracy:",acc)
	accArr=sorted(accArr,key=itemgetter(1),reverse=True)
	print("Max Accuracy=",accArr[0][1],"For k=",accArr[0][0])

main()