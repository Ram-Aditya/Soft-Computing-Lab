import csv 
import sys 

from operator import itemgetter


import math

from random import shuffle
def takeSecond(elem):
    return elem[1]

class KNN:
    def __init__(self):
        self.distance = []
        self.data = []
        self.data_count = 0

    def ReadData(self,filename):
        with open(filename,'r') as csvfile:
            csvreader = csv.reader(csvfile)
            count = 0 
            for row in csvreader:
                # print(row)
                if( count != 0 ):
                    self.data.append(row)
                if( row[-1] == 'Yes' or row[-1] == 'Iris-setosa' ):
                    row[-1] = 1.0
                elif( row[-1] == 'No' or row[-1] == 'Iris-versicolor' ):
                    row[-1] = 0.0
                count += 1
        shuffle(self.data)
        # shuffle(self.data)
        # print(self.data)
        self.data_count = len(self.data)
    
    def get_distance(self,row_index,test_index):
        dist = 0
        for i in range(len(self.data[row_index]) - 1):
            dist += pow( (float(self.data[row_index][i]) - float(self.data[test_index][i]) ),2)
        #dist = math.sqrt(dist) 
        self.distance.append( (row_index, dist))

    def update_distance(self,train_start_index,train_end_index,test_index):
        self.distance = []
        row_index = train_start_index 
        while row_index != train_end_index:
            self.get_distance(row_index,test_index)
            row_index = ( row_index + 1 )%self.data_count
        # print(self.distance)
        # return
        
    def implementation(self,test_start_index,test_end_index,train_start_index,train_end_index,k):
        tp = fn = fp = tn = 0
        ans_list = []
        test_index  = test_start_index
        while test_index < test_end_index:
            self.update_distance(train_start_index,train_end_index,test_index)
            self.distance=sorted(self.distance,key=itemgetter(1))
            yes_count = 0 
            no_count = 0 
            if k < len(self.distance):
                i = 0 
                while i < k:
                    if self.data[ self.distance[i][0] ][-1] == 1.0:
                        yes_count += 1 
                    elif self.data[ self.distance[i][0] ][-1] == 0.0:
                        no_count += 1
                    i += 1
            if yes_count > no_count :
                predicted_output = 1.0
            else:
                predicted_output = 0.0
            if( predicted_output == 1 and self.data[test_index][-1] == 1 ):
                tp += 1
            elif( predicted_output == 0 and self.data[test_index][-1] == 1 ):
                fn += 1
            elif( predicted_output == 1 and self.data[test_index][-1] == 0):
                fp += 1
            elif( predicted_output == 0 and self.data[test_index][-1] == 0):
                tn += 1
            test_index = (test_index + 1)%self.data_count
        error_num = fp+fn
        count = self.data_count
        test_len = count - 0.9*count
        ans_list.append(1 - error_num/(test_len))
        if tp != 0 or fp != 0 :
            ans_list.append((tp*100/(tp+fp))/(test_len))
        ans_list.append((tp*100/(tp+fn))/(test_len))
        if tn != 0 or fn != 0 :
            ans_list.append((tn*100/(tn+fn))/(test_len))
        # print("kiki") 
        if tn != 0 or fp != 0 :
            ans_list.append((tn*100/(tn+fp))/(test_len)) 
        # print("Acc  : " ,ans_list[0]*100)
        # if tp != 0 or fp != 0 :
        #     print("Percision(+) : {}".format( tp*100/(tp+fp)))
        # if tp != 0 or fn != 0 :
        #     print("Recall(+) : {}".format(tp*100/(tp+fn)))
        # if tn != 0 or fn != 0 :
        #     print("Percision(-) : {}".format(tn*100/(tn+fn)))
        # if tn != 0 or fp != 0 :
        #     print("Recall(-) : {}".format(tn*100/(tn+fp)))
        return ans_list


        
    def K_fold_training(self,fold_value,k):
        train_start_index = 0
        # train_start_index = 0 
        # train_end_index = fold_range
        # test_start_index = int(train_end_index)%count
        # test_end_index = (test_start_index+int(0.1*count) )%(count)
        train_end_index = fold_value*self.data_count
        test_start_index = int(train_end_index)%self.data_count
        test_end_index = (test_start_index + int(( 1 - fold_value )*self.data_count ))%self.data_count
        fold_count = 0
        Acc = [0.0 for _ in range(5)]
        while fold_count < 10:
            # self.train(int(train_start_index),int(train_end_index))
            # print("Fold : " , fold_count + 1)
            # print(train_start_index," train : ",train_end_index)
            # print(test_start_index, " test : ",test_end_index)

            ans_list = self.implementation(int(test_start_index),int(test_end_index),int(train_start_index),int(train_end_index),k)
            # print()
            for i in range(len(ans_list)):
                Acc[i] += ans_list[i]
            # return
            test_start_index = (test_end_index + 1)%self.data_count
            test_end_index = test_end_index = (test_start_index + ( 1 - fold_value )*self.data_count )%self.data_count
            train_start_index = (test_end_index )%self.data_count
            train_end_index = ( train_start_index + fold_value*self.data_count)%self.data_count
            fold_count += 1
        #     print("------------------------------------------------------------------------------------")
        print("Total Accuracy : ",Acc[0]*100/10)
        # print("Average pricision(+) : ",Acc[1])
        # print("Average Recall(+) : ",Acc[2])
        # print("Average pricision(-) : ",Acc[3])
        # print("Average Recall(-) : ",Acc[4])
        return Acc[0]*10

def main():
    K  = KNN()
    filename  = input("Enter file name : ")
    K.ReadData(filename)
    fold_value = 0.9
    # value_of_k = [ i for i in range(1,15,1) ]
    value_of_k = [ i for i in range(1,16,1)]
    max_k = 0
    max_acc = 0
    for k in value_of_k :
        print("For k = ",k)
        acc1 = K.K_fold_training(fold_value,k)
        if( acc1 > max_acc ):
            max_k = k
            max_acc = acc1

    print("Maximum Accuracy : ",max_acc," for k = ",max_k)
if __name__ == '__main__':
    main()