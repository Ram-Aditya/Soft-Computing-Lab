## Merge  my PR
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

#Activation function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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

class NeuralNetwork:
	def __init__(self,noLayers):
		self.noLayers=noLayers
		self.weightsB2FLayer={}
		self.biasLayer={}
		self.weightedOpLayer={}
		self.activatedOpLayer={}
		self.deltaLayer={}
		self.noNodesLayer=None

	def createNetwork(self,noNodesLayer):
		self.noNodesLayer=noNodesLayer
		for i in range(self.noLayers-1):
			self.weightsB2FLayer[i], self.biasLayer[i], self.deltaLayer[i], self.weightedOpLayer[i], self.activatedOpLayer[i]= self.createLayer(noNodesLayer[i],noNodesLayer[i+1])
		self.weightedOpLayer[self.noLayers-1],self.activatedOpLayer[self.noLayers-1], self.biasLayer[self.noLayers-1]= self.createOuputLayer(noNodesLayer[self.noLayers-1])

	def createOuputLayer(self,no_Nodes):
		weightedOp=[0 for _ in range(no_Nodes)]
		activatedOp=[0 for _ in range(no_Nodes)]
		bias=[1/6 for _ in range(no_Nodes)]
		return weightedOp, activatedOp , bias

	def createLayer(self,no_Nodes,no_NodesNext):
		weightsMat=[[1/(no_Nodes*no_NodesNext) for _ in range(no_NodesNext)] for _ in range(no_Nodes)]
		bias=[0 for _ in range(no_Nodes)]
		delta=[0 for _ in range(no_NodesNext)]
		weightedOp=[0 for _ in range(no_Nodes)]
		activatedOp=[0 for _ in range(no_Nodes)]
		return weightsMat, bias, delta, weightedOp, activatedOp

def calOpMat(weightsMat,biasNext,activatedOp):
	activatedOpNext=[]
	noCurrNodes=len(weightsMat)
	noNextNodes=len(weightsMat[0])
	for nextNode in range(noNextNodes):
		weightedSum=0
		for currNode in range(noCurrNodes):
			weightedSum+=(float(activatedOp[currNode])*weightsMat[currNode][nextNode])
		activatedOpNext.append(sigmoid(weightedSum+biasNext[nextNode]))
	return activatedOpNext

def forwardPass(data,network):
	network.activatedOpLayer[0]=data
	activatedOp=data
	for layer in range(network.noLayers-1):
		activatedOp=calOpMat(network.weightsB2FLayer[layer],network.biasLayer[layer+1],activatedOp)
		network.activatedOpLayer[layer+1]=activatedOp

def calDelta(network,train):
	n=network.noLayers
	currOp=network.activatedOpLayer[n-1]
	for currNode in range(network.noNodesLayer[n-1]):
		errorCorr=(dataSet[train][totalFields-1]-currOp[currNode])*(1-currOp[currNode])*currOp[currNode]
		network.deltaLayer[n-2][currNode]=errorCorr
	#print(n-2)
	for layer in range(n-2,0,-1):
		#print("dfd")
		currOp=network.activatedOpLayer[layer]
		for currNode in range(network.noNodesLayer[layer]):
			errorCorr=(1-currOp[currNode])*currOp[currNode]
			delta=0
			currWtMat=network.weightsB2FLayer[layer]
			currDeltaArr=network.deltaLayer[layer]
			deltaNext=0
			for nextNode in range(network.noNodesLayer[layer+1]):
				deltaNext+=(currWtMat[currNode][nextNode]*currDeltaArr[nextNode])
			#print(layer-1)
			network.deltaLayer[layer-1][currNode]=errorCorr*deltaNext

def correctWeights(network):
	for layer in range(network.noLayers-1):
		no_fwd=network.noNodesLayer[layer+1]
		no_curr=network.noNodesLayer[layer]
		for i in range(no_fwd):
			delta=network.deltaLayer[layer][i]
			for j in range(no_curr):
				network.weightsB2FLayer[layer][j][i]+=(learningRate*delta*(network.activatedOpLayer[layer][j]))
		if(layer!=0):
			for i in range(no_curr):
					network.biasLayer[layer][i]+=(learningRate*network.deltaLayer[layer-1][i])
	for opNode in range(network.noNodesLayer[network.noLayers-1]):
		network.biasLayer[network.noLayers-1][opNode]+=(learningRate*network.deltaLayer[network.noLayers-2][opNode])

def trainModelOnce(dataSet,network,train_st):
	train=train_st
	trainNo=0
	while(trainNo<no_TrainSets):
		forwardPass(dataSet[train],network)
		predOp=network.activatedOpLayer[network.noLayers-1][0]
		correctOp=dataSet[train][totalFields-1]
		#if(predOp>=0.5):
		#	predOp=1
		#else:
		#	predOp=0
		#if(predOp!=correctOp):
		#	totalErrors+=1
		calDelta(network,train)
		correctWeights(network)
		train=(train+1)%totalSets
		trainNo+=1
	return

def trainModel(dataSet,network,train_st):
	iterations=0
	#totalErrors=1
	while(iterations<MAX_ITERATIONS):
		trainModelOnce(dataSet,network,train_st)
		iterations+=1
	return

def testModel(dataSet,network,test_st):
	falsePos=truePos=falseNeg=trueNeg=posCount=negCount=0
	testNo=0
	test=test_st
	while testNo<no_TrainSets:
		forwardPass(dataSet[test],network)
		predOp=network.activatedOpLayer[network.noLayers-1][0]
		if(predOp>=(0.5)):
			predOp=1
		else:
			predOp=0
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
def trainAndCrossValidate(dataSet,network):
	kFold=0
	train_st=0	
	net_precisionPos=net_precisionNeg=net_recallPos=net_recallNeg=0
	net_precisionPosCount=net_precisionNegCount=net_recallPosCount=net_recallNegCount=0
	net_accuracy=net_error=0

	while(kFold<10):

		print("Fold No.",kFold+1)
		print("Training model...")
		trainModel(dataSet,network,train_st)
		print("Testing model...")
		test_st=(train_st+no_TrainSets)%totalSets
		truePos,falsePos,trueNeg,falseNeg=testModel(dataSet,network,test_st)

		if((truePos+falsePos)!=0):
			precisionPos=(truePos)/(truePos+falsePos)
			net_precisionPos+=(precisionPos)
			net_precisionPosCount+=1
			print("Precision_Pos %=",precisionPos*100,end='\t')
		if((truePos+falseNeg)!=0):
			recallPos=(truePos)/(truePos+falseNeg)
			net_recallPos+=(recallPos)
			net_recallPosCount+=1
			print("Recall_Pos %=",recallPos*100)
		if((trueNeg+falseNeg)!=0):
			precisionNeg=(trueNeg)/(trueNeg+falseNeg)
			net_precisionNeg+=(precisionNeg)
			net_precisionNegCount+=1
			print("Precision_Neg %=",precisionNeg*100,end='\t')
		if((trueNeg+falsePos)!=0):
			recallNeg=(trueNeg)/(trueNeg+falsePos)
			net_recallNeg+=(recallNeg)
			net_recallNegCount+=1
			print("Recall_Neg %=",recallNeg*100)
		accuracy=((trueNeg+truePos)/(trueNeg+truePos+falseNeg+falsePos))
		net_accuracy+=accuracy
		print("Accuracy %=",accuracy*100)
		train_st=(train_st+setSize)%totalSets
		kFold+=1
		print("***************************************************")


	net_precisionPos/=net_precisionPosCount
	net_recallPos/=net_recallPosCount
	net_precisionNeg/=net_recallNegCount
	net_recallNeg/=net_recallNegCount
	net_accuracy/=10
	print("Average Recall_Pos %=",net_recallPos*100,"\nAverage Precision_Pos %=",net_precisionPos*100)
	print("Average Recall_Neg %=",net_recallNeg*100,"\nAverage Precision_Neg %=",net_precisionNeg*100)
	print("Accuracy %=",net_accuracy*100)

def main():
	global dataSet,totalAttr,totalFields
	filename=sys.argv[1]
	fields, dataSet=parseCSV(filename)
	shuffle(dataSet)

	network=NeuralNetwork(3)
	noNodesLayer=[totalAttr,5,1]

	network.createNetwork(noNodesLayer)
	trainAndCrossValidate(dataSet,network)

main()
