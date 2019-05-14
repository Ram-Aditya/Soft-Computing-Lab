import csv
import sys
from random import shuffle
import math

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
discreet=[]

#Parse dataset
def parseCSV(filename,attrChrom):

	fields = []
	rows = []
	chromFields=[]
	global totalAttr,totalFields,totalSets,no_TrainSets,no_TestSets,setSize

	chromAttrCount=0
	for i in attrChrom:
		if(i==1):
			chromAttrCount+=1
	chrom_totalFields=chromAttrCount+1

	with open(filename, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		fields = next(csvreader)
		totalFields=len(fields)
		totalAttr=totalFields-1
		for i in range(len(fields)-1):
			if attrChrom[i]==1:
				chromFields.append(fields[i])
		chromFields.append(fields[totalFields-1])
		nameFlag=1
		className=""
		questFlag=0
		#i=0
		for row in csvreader:
			#i+=1
			chromRow=[]
			# if(i>5000):
			# 	break
			# for attr in row:
			# 	if attr=="?":
			# 		questFlag=1
			# 		break
			# if(questFlag):
			# 	questFlag=0
			# 	continue
			if(nameFlag):
				className=row[totalFields-1]
				nameFlag=0
			for i in range(len(row)-1):
				if attrChrom[i]==1:
					chromRow.append(row[i])
			if row[totalFields-1]==className:
				chromRow.append(1)
			else:
				chromRow.append(0)
			rows.append(chromRow)

	for row in rows:
	 	for i in range(len(row)):
	 		if(i!=(len(row)-1)):
	 			row[i]=float(row[i])

	totalSets=len(rows)
	setSize=(totalSets)//10
	no_TrainSets=9*setSize
	no_TestSets=totalSets-no_TrainSets
	#print(rows)
	return chromFields,rows

#	def gaussProb(traiattr,variance,mean):
# 	root2=math.sqrt(2)
# 	for attr in range(totalAttr):
def getMean(train_st):
	mean=[[0,0] for _ in range(totalAttr)]
	classCount=[0,0]
	trainNo=0
	row=train_st
	while trainNo<no_TrainSets:
		for attr in range(totalAttr):
			if(dataSet[row][-1]==0):
				mean[attr][0]+=(dataSet[row][attr])
				classCount[0]+=1
			else:
				mean[attr][1]+=(dataSet[row][attr])
				classCount[1]+=1
		row=(row+1)%totalSets
		trainNo+=1
	for attr in range(totalAttr):
		mean[attr][0]/=classCount[0]
		mean[attr][1]/=classCount[1]
	return mean,classCount

def getVariance(train_st,mean,classCount):
	variance=[[0,0] for _ in range(totalAttr)]
	trainNo=0
	row=train_st
	while trainNo<no_TrainSets:
		for attr in range(totalAttr):
			if(dataSet[row][-1]==0):
				variance[attr][0]+=((mean[attr][0]-dataSet[row][attr])*(mean[attr][0]-dataSet[row][attr]))
			else:
				variance[attr][1]+=((mean[attr][1]-dataSet[row][attr])*(mean[attr][1]-dataSet[row][attr]))
		row=(row+1)%totalSets
		trainNo+=1
	for attr in range(totalAttr):
		variance[attr][0]/=(classCount[0]-1)
		variance[attr][1]/=(classCount[1]-1)
	return variance

def gaussProb(var,mean,x):
	if(var==0):
		var=0.000000000000000000001
	return (1/(math.sqrt(2*var)*math.pi))*math.exp(-( (x-mean)*(x-mean) )/(2*var) )

def predictCont(test_i,mean,variance,classCount):
	probC1=probC2=1
	countC1=classCount[0]
	countC2=classCount[1]
	for attr in range(totalAttr):
		probC1*=(gaussProb(variance[attr][0],mean[attr][0],dataSet[test_i][attr]))
		probC2*=(gaussProb(variance[attr][1],mean[attr][1],dataSet[test_i][attr]))
	#print(probC1,probC2)
	if(probC1>probC2):
		return 0
	else:
		return 1

def getNumAttr_Lab(train_st):
	dictArr=[{} for _ in range(totalAttr)]
	classCount=[0,0]
	trainNo=0
	row=train_st
	while trainNo<no_TrainSets:
		for attr in range(totalAttr):
				if(dataSet[row][attr] not in dictArr[attr]):
					dictArr[attr][dataSet[row][attr]]=[0,0]
				if(dataSet[row][-1]==0):
					dictArr[attr][dataSet[row][attr]][0]+=1
					classCount[0]+=1
				else:
					dictArr[attr][dataSet[row][attr]][1]+=1
					classCount[1]+=1
		row=(row+1)%totalSets
		trainNo+=1
	return dictArr, classCount

def predictDiscreet(test_i,numAttr_Class,classCount):
	probC1=probC2=0
	countC1=classCount[0]
	countC2=classCount[1]
	probC1=countC1/(countC1+countC2)
	probC2=countC2/(countC1+countC2)
	for attr in range(totalAttr):
		attrVal=dataSet[test_i][attr]
		probC1*=(numAttr_Class[attr][attrVal][0]/countC1)
		attrVal=dataSet[test_i][attr]
		probC2*=(numAttr_Class[attr][attrVal][1]/countC2)

	if(probC1>probC2):
		return 0
	else:
		return 1

def testModel(test_st,numAttr_Class=None,classCount=None,variance=None,mean=None):
	falsePos=truePos=falseNeg=trueNeg=posCount=negCount=0
	testNo=0
	test=test_st
	while testNo<no_TrainSets:
		predOp=predictDiscreet(test,numAttr_Class,classCount)
		#predOp=predictCont(test,mean,variance,classCount)
		correctOp=dataSet[test][-1]
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
		#For training set
		numAttr_Class,classCount=getNumAttr_Lab(train_st)
		#mean,classCount=getMean(train_st)
		#variance=getVariance(train_st,mean,classCount)
		#For testing set 
		test_st=(train_st+no_TrainSets)%totalSets
		truePos,falsePos,trueNeg,falseNeg=testModel(test_st=test_st,numAttr_Class=numAttr_Class,classCount=classCount,variance=None,mean=None)
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
	return net_accuracy*100
	# print("Average Recall_Pos %=",net_recallPos*100,"\nAverage Precision_Pos %=",net_precisionPos*100)
	# print("Average Recall_Neg %=",net_recallNeg*100,"\nAverage Precision_Neg %=",net_precisionNeg*100)
	# print("Accuracy %=",net_accuracy*100)

def initDiscreetBool():
	discreet=[False for _ in range(totalAttr)]

def naive(filename,attrChrom):
	global dataSet,totalAttr,totalFields
	#filename=sys.argv[1]
	fields, dataSet=parseCSV(filename,attrChrom)
	#shuffle(dataSet)
	#initDiscreetBool()
	return trainAndCrossValidate()

print(naive("SPECT_SHUFFLE.csv",[1 for _ in range(22)]))