# -*- coding: utf-8 -*-
"""MFGRO (Modified).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15uWrOL95u1KD1qc0XDIV56WQsmUGb6b2
"""

# Commented out IPython magic to ensure Python compatibility.
####################################### Hybrid Feature Selection ###########################################
#Import required libraries
import numpy as np
import pandas as pd
import random
import math,time,sys
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
# %matplotlib inline
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier

#convert to probability

def sigmoid1(gamma):                #S-shaped transfer function 1
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid2(gamma):                #S-shaped transfer function 2
	gamma /= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))
		
def sigmoid3(gamma):               #S-shaped transfer function 3
	gamma /= 3
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid4(gamma):              #S-shaped transfer function 4
	gamma *= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))


def Vfunction1(gamma):              #Vshaped transer function V1
	return abs(np.tanh(gamma))

def Vfunction2(gamma):               #Vshaped transer function V2
	val = (math.pi)**(0.5)
	val /= 2
	val *= gamma
	val = math.erf(val)
	return abs(val)

def Vfunction3(gamma):                #Vshaped transer function V3
	val = 1 + gamma*gamma
	val = math.sqrt(val)
	val = gamma/val
	return abs(val)

def Vfunction4(gamma):                 #Vshaped transer function V4
	val=(math.pi/2)*gamma
	val=np.arctan(val)
	val=(2/math.pi)*val
	return abs(val)


def fitness(position):                     #fitness calculation 
	cols=np.flatnonzero(position)
	val=1
	if np.shape(cols)[0]==0:
		return val	
	# clf = RandomForestClassifier(n_estimators=300) #Different classifiers
	clf=KNeighborsClassifier(n_neighbors=5)
	# clf=MLPClassifier( alpha=0.01, max_iter=1000) #hidden_layer_sizes=(1000,500,100)
	#cross=3
	#test_size=(1/cross)
	#X_train, X_test, y_train, y_test = train_test_split(trainX, trainy,  stratify=trainy,test_size=test_size)
	train_data=trainX[:,cols]
	test_data=testX[:,cols]
	clf.fit(train_data,trainy)
	val=1-clf.score(test_data,testy)

	#in case of multi objective  []
	set_cnt=sum(position)
	set_cnt=set_cnt/np.shape(position)[0]
	val=omega*val+(1-omega)*set_cnt
	return val

def onecount(position):                        #number of ones count 
	cnt=0
	for i in position:
		if i==1.0:
			cnt+=1
	return cnt


def allfit(population):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i])     
		#print('acc: ', acc[i])
	return acc

def initialize(popSize,dim):                      #popultation initialization 
	population=np.zeros((popSize,dim))
	minn = 1
	maxx = math.floor(0.8*dim)
	if maxx<minn:
		minn = maxx
	
	for i in range(popSize):
		random.seed(i**3 + 10 + time.time() ) 
		no = random.randint(minn,maxx)
		if no == 0:
			no = 1
		random.seed(time.time()+ 100)
		pos = random.sample(range(0,dim-1),no)
		for j in pos:
			population[i][j]=1
		
		# print(population[i])  
		
	return population

def toBinary(population,popSize,dimension,oldPop):          #Binary conversion

	for i in range(popSize):
		for j in range(dimension):
			temp = Vfunction3(population[i][j])

			# if temp > 0.5: # sfunction
			# 	population[i][j] = 1
			# else:
			# 	population[i][j] = 0

			if temp > 0.5: # vfunction
				population[i][j] = (1 - oldPop[i][j])
			else:
				population[i][j] = oldPop[i][j]
    
	return population

def toBinary(solution,dimension):
	# print("continuous",solution)
	Xnew = np.zeros(np.shape(solution))
	for i in range(dimension):
		temp = Vfunction3(abs(solution[i]))

		random.seed(time.time()+i)
		if temp > random.random(): # sfunction
			Xnew[i] = 1
		else:
			Xnew[i] = 0
		# if temp > 0.5: # vfunction
		# 	Xnew[i] = 1 - abs(solution[i])
		# else:
		# 	Xnew[i] = abs(solution[i])
	# print("binary",Xnew)
	return Xnew

def toBinaryX(solution,dimension,oldsol,trainX, testX, trainy, testy):
	Xnew = np.zeros(np.shape(solution))
	Xnew1 = np.zeros(np.shape(solution))
	Xnew2 = np.zeros(np.shape(solution))
	for i in range(dimension):
		temp = sigmoid1(abs(solution[i]))
		random.seed(time.time()+i)
		r1 = random.random()
		if temp > r1: # sfunction
			Xnew1[i] = 1
		else:
			Xnew1[i] = 0

		temp = sigmoid1i(abs(solution[i]))
		if temp > r1: # sfunction
			Xnew2[i] = 1
		else:
			Xnew2[i] = 0

	fit1 = fitness(Xnew1,trainX,testX,trainy,testy)
	fit2 = fitness(Xnew2,trainX,testX,trainy,testy)
	fitOld =  fitness(oldsol,trainX,testX,trainy,testy)
	if fit1<fitOld or fit2<fitOld:
		if fit1 < fit2:
			Xnew = Xnew1.copy()
		else:
			Xnew = Xnew2.copy()
	return Xnew
	# else: CROSSOVER
	Xnew3 = Xnew1.copy()
	Xnew4 = Xnew2.copy()
	for i in range(dimension):
		random.seed(time.time() + i)
		r2 = random.random()
		if r2>0.5:
			tx = Xnew3[i]
			Xnew3[i] = Xnew4[i]
			Xnew4[i] = tx
	fit1 = fitness(Xnew3,trainX,testX,trainy,testy)
	fit2 = fitness(Xnew4,trainX,testX,trainy,testy)
	if fit1<fit2:
		return Xnew3
	else:
		return Xnew4
	# print("binary",Xnew)

omega = 0.85          #weightage for no of features and accuracy
popSize = 20
max_iter = 30
S = 2


data = pd.read_csv('/content/drive/MyDrive/COVID-CT/ResNet_78%.csv')             # Data load
label = pd.read_csv('/content/drive/MyDrive/COVID-CT/labels(covidCT).csv')       # Label load
data = np.asarray(data)
label = label['class']
label = np.asarray(label)
(a,b)=np.shape(data)
print(a,b)
dimension = np.shape(data)[1]                                                    #particle dimension




best_accuracy = -1
best_no_features = -1
average_accuracy = 0
global_count = 0
accuracy_list = []
features_list = []

##### KFold Validations ########

from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
Fold = 5

kfold = KFold(Fold, True, 1)

f = 0

for train_index, test_index in kfold.split(data):
  trainX,  trainy= np.asarray(data[train_index]), np.asarray(label[train_index])
  testX, testy  = np.asarray(data[test_index]), np.asarray(label[test_index])
  for train_iteration in range(11):
    print("\nFold: {}, Iteration : {}".format(f+1,train_iteration+1))

#cross = 5
    #test_size = (1/cross)
    #trainX, testX, trainy, testy = train_test_split(data, label,test_size = 0.2)#stratify=label ,test_size=test_size)


    # clf = RandomForestClassifier(n_estimators=300)
    clf=KNeighborsClassifier(n_neighbors=5)
    # clf=MLPClassifier(alpha=0.001, max_iter=1000) #hidden_layer_sizes=(1000,500,100)
    clf.fit(trainX,trainy)
    val=clf.score(testX,testy)
    whole_accuracy = val
    print("Total Acc: ",val)

    # for population_iteration in range(2):
    global_count += 1
    print('global: ',global_count)

    x_axis = []
    y_axis = []

    population = initialize(popSize,dimension)
    # print(population)

    start_time = datetime.now()
    fitList = allfit(population)
    bestInx = np.argmin(fitList)
    fitBest = min(fitList)
    Mbest = population[bestInx].copy()
    for currIter in range(max_iter):

      popnew = np.zeros((popSize,dimension))
      x_axis.append(currIter)
      y_axis.append(min(fitList))
      for i in range(popSize):
        random.seed(time.time() + 10.01)
        randNo = random.random()
        if randNo<0.5 :
          #chain foraging
          random.seed(time.time())
          r = random.random()
          alpha = 2*r*(abs(math.log(r))**0.5)
          if i == 1:
            popnew[i] = population[i] + r * (Mbest - population[i]) + alpha*(Mbest - population[i])
          else:
            popnew[i] = population[i] + r * (population[i-1] - population[i]) + alpha*(Mbest - population[i])
        else:
          #cyclone foraging
          cutOff = random.random()
          r = random.random()
          r1 = random.random()
          beta = 2 * math.exp(r1 * (max_iter - currIter + 1) / max_iter) * math.sin(2 * math.pi * r1)
          if currIter/max_iter < cutOff:
            # exploration
            Mrand = np.zeros(np.shape(population[0]))
            no = random.randint(1,max(int(0.1*dimension),2))
            random.seed(time.time()+ 100)
            pos = random.sample(range(0,dimension-1),no)
            for j in pos:
              Mrand[j] = 1

            if i==1 :
              popnew[i] = Mrand + r * (Mrand - population[i]) + beta * (Mrand - population[i])
            else:
              popnew[i] = Mrand + r * (population[i-1] - population[i]) + beta * (Mrand - population[i])
          else:
            # exploitation
            if i == 1:
              popnew[i] = Mbest + r * (Mbest - population[i]) + beta * (Mbest - population[i])
            else:
              popnew[i] = Mbest + r * (population[i-1] - population[i]) + beta * (Mbest - population[i])

      # print(popnew)
      
      popnew = toBinary(popnew,popSize,dimension,population)
      #print('popnew: ', popnew.shape)
      popnewTemp = popnew.copy()
      #compute fitness for each individual
      fitList = allfit(popnew)
      #print('fitlist: ', len(fitList))
      if min(fitList)<fitBest :
        bestInx = np.argmin(fitList)
        fitBest = min(fitList)
        Mbest = popnew[bestInx].copy()
      #print('fitList,fitBest: ',fitList,fitBest)

      #somersault foraging
      for i in range(popSize):
        r2 = random.random()
        random.seed(time.time())
        r3 = random.random()
        popnew[i] = popnew[i] + S * (r2*Mbest - r3*popnew[i])

      popnew = toBinary(popnew,popSize,dimension,popnewTemp)
      #compute fitness for each individual
      fitList = allfit(popnew)
      #print('fitList: ', fitList)
      if min(fitList)<fitBest :
        bestInx = np.argmin(fitList)
        print('BestInx: ',bestInx)
        fitBest = min(fitList)
        Mbest = popnew[bestInx].copy()
        print('Mbest: ', Mbest)
        print('Mbest_shape: ', len(Mbest))
      # print(fitList,fitBest)
      ############# Feature Selection ############
      ############################################
      
      Binary_featrue_set = []
      Accuracies = []
      for i in range (len(fitList)):
        pop = popnew[i].copy()
        print('pop: ', pop)
        features = pop.copy()
        cols_=np.flatnonzero(features)
        X_test_pop=testX[:,cols_]
        X_train_pop=trainX[:,cols_]
        #print(np.shape(feature))

        # clf = RandomForestClassifier(n_estimators=300)
        clf=KNeighborsClassifier(n_neighbors=5)
        #clf=MLPClassifier( alpha=0.001, max_iter=2000) #hidden_layer_sizes=(1000,500,100 ),
        if onecount(features) != 0 :
          Binary_featrue_set.append(features)
          clf.fit(X_train_pop,trainy)
          val_pop=clf.score(X_test_pop, testy )
          Accuracies.append(val_pop)
          #print(val_pop,onecount(features))


      population = popnew.copy()


    time_required = datetime.now() - start_time

    # pyplot.plot(x_axis,y_axis)
    # pyplot.xlim(0,max_iter)
    # pyplot.ylim(max(0,min(y_axis)-0.1),min(max(y_axis)+0.1,1))
    # pyplot.show()


    output = Mbest.copy()
    print('output: ',output)

    #test accuracy
    cols=np.flatnonzero(output)
    #print(cols)
    X_test=testX[:,cols]
    X_train=trainX[:,cols]
    #print(np.shape(feature))

    # clf = RandomForestClassifier(n_estimators=300)
    clf=KNeighborsClassifier(n_neighbors=5)
    #clf=MLPClassifier( alpha=0.001, max_iter=2000) #hidden_layer_sizes=(1000,500,100 ),
    clf.fit(X_train,trainy)
    val=clf.score(X_test, testy )
    print(val,onecount(output))
    '''
    # classifier
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    y_score = clf.fit((X_train,trainy).decision_function(X_test))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(testy[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        #plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve for class {}(area = {:.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
    plt.show()
    '''

    accuracy_list.append(val)
    features_list.append(onecount(output))
    if ( val == best_accuracy ) and ( onecount(output) < best_no_features ):
      best_accuracy = val
      best_no_features = onecount( output )
      best_time_req = time_required
      best_whole_accuracy = whole_accuracy

    if val > best_accuracy :
      best_accuracy = val
      best_no_features = onecount( output )
      best_time_req = time_required
      best_whole_accuracy = whole_accuracy
      # classifier
      clf = OneVsRestClassifier(LinearSVC(random_state=0))
      y_score = clf.fit(X_train,trainy)

      # Compute ROC curve and ROC area for each class
      '''
      fpr = dict()
      tpr = dict()
      roc_auc = dict()
      n_classes=2
      for i in range(n_classes):
          fpr[i], tpr[i], _ = roc_curve(testy[:, i], y_score[:, i])
          roc_auc[i] = auc(fpr[i], tpr[i])
          print(roc_auc[i])

      # Plot of a ROC curve for a specific class
      for i in range(n_classes):
          #plt.figure()
          plt.plot(fpr[i], tpr[i], label='ROC curve for class {}(area = {:.2f})'.format(i, roc_auc[i]))
          plt.plot([0, 1], [0, 1], 'k--')
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.05])
          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.title('Receiver operating characteristic')
          plt.legend(loc="lower right")
      plt.show()
      '''

  #print('best: ',best_accuracy, best_no_features)
  f += 1
# print('avg: ',average_accuracy/10)


# accuracy_list = np.array(accuracy_list)
# accuracy_list.sort()
# accuracy_list = accuracy_list[-4:]
# average = np.mean(accuracy_list)
# stddev = np.std(accuracy_list)

# accuracy_list = list(accuracy_list)
# avgFea = 0
# for i in accuracy_list:
# 	inx = accuracy_list.index(i)
# 	avgFea += features_list[inx]
# avgFea /= 4
'''
temp=sys.argv[1].split('/')[-1]
with open("../Result/result_MRFOv3_uci20.csv","a") as f:
	print(temp,"%.2f" % (100*best_whole_accuracy) ,
		np.shape(df)[1] - 1,"%.2f" % (100*best_accuracy),best_no_features,file=f)

'''
def goldenratiomethod(train_index, test_index, popSize, maxIter):

	#---------------------------------------------------------------------
	
	'''
	df=pd.read_csv(dataset)
	(a,b)=np.shape(df)
	print(a,b)
	data = df.values[:,0:b-1]
	label = df.values[:,b-1]
	dimension = np.shape(data)[1] #particle dimension
	'''

	#---------------------------------------------------------------------

	trainX,  trainy= np.asarray(data[train_index]), np.asarray(label[train_index])
	testX, testy  = np.asarray(data[test_index]), np.asarray(label[test_index])


	clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(trainX,trainy)
	val=clf.score(testX,testy)
	whole_accuracy = val
	print("Total Acc: ",val)

	x_axis = []
	y_axis = []
	population = initialize(popSize,dimension)
	BESTANS = np.zeros(np.shape(population[0]))
	BESTACC = 1000

	start_time = datetime.now()
	population_list = []
	accuracies = []
	for currIter in range(1,maxIter):

		fitList = allfit(population,trainX,testX,trainy,testy)
		y_axis.append(min(fitList))
		x_axis.append(currIter)
		worstInx = np.argmax(fitList)
		fitWorst = max(fitList)
		Xworst = population[worstInx].copy()

		Xave = population.sum(axis=0)
		Xave = np.divide(Xave,popSize)
		# for x in Xave:
		# 	print("%.2f"%x,end=',')
		# print()
		XaveBin= output
		FITave = fitness(XaveBin, trainX, testX, trainy, testy)
		if FITave<fitWorst:
			population[worstInx] = XaveBin.copy()
			fitList[worstInx] = FITave
		


		for i in range(popSize):
			Xi = population[i].copy()
			j = i
			while j == i:
				random.seed(time.time()+j)
				j = random.randint(0, popSize-1)
			Xj = population[j].copy()
			FITi = fitList[i]
			FITj = fitList[j]

			Xave = population.sum(axis=0)
			Xave = np.subtract(Xave,population[i])
			Xave = np.subtract(Xave,population[j])
			Xave = np.divide(Xave,(popSize-2))
			XaveBin = toBinary(Xave,dimension)
			FITave = fitness(XaveBin, trainX, testX, trainy, testy)
			# print(i,j,FITi,FITj,FITave)
			Xbest = np.zeros(np.shape(Xi))
			Xmedium = np.zeros(np.shape(Xi))
			Xworst = np.zeros(np.shape(Xi))
			
			if FITi < FITj < FITave:
				Xbest = Xi.copy()
				Xmedium = Xj.copy()
				Xworst = Xave.copy()
			elif FITi < FITave < FITj:
				Xbest = Xi.copy()
				Xmedium = Xave.copy()
				Xworst = Xj.copy()
			elif FITj < FITi < FITave:
				Xbest = Xj.copy()
				Xmedium = Xi.copy()
				Xworst = Xave.copy()
			elif FITj < FITave < FITi:
				Xbest = Xj.copy()
				Xmedium = Xave.copy()
				Xworst = Xi.copy()
			elif FITave < FITi < FITj:
				Xbest = Xave.copy()
				Xmedium = Xi.copy()
				Xworst = Xj.copy()
			elif FITave < FITj < FITi:
				Xbest = Xave.copy()
				Xmedium = Xj.copy()
				Xworst = Xi.copy()

				Xt = np.subtract(Xmedium,Xworst)
				T = currIter/maxIter
				Ft = (golden/(5**0.5)) * (golden**T - (1 - golden)**T)
				random.seed(19*time.time() + 10.01)
				Xnew = np.multiply(Xbest,(1-Ft)) + np.multiply(Xt,random.random()*Ft)
				Xnew = toBinaryX(Xnew,dimension,population[i],trainX, testX, trainy, testy)
				#print('Xnew: ', Xnew )
				FITnew = fitness(Xnew, trainX, testX, trainy, testy)
				#print('FITnew: ', FITnew)
				# if FITnew < fitList[i]:
					# print(i,j,"updated2")
				population[i] = Xnew.copy()
				fitList[i] = FITnew

		#second phase
		worstInx = np.argmax(fitList)
		fitWorst = max(fitList)
		Xworst = population[worstInx].copy()
		bestInx = np.argmin(fitList)
		fitBest = min(fitList)
		Xbest = population[bestInx].copy()
		for i in range(popSize):
			Xi = population[i].copy()
			#print('Xi: ', Xi)
			random.seed(29*time.time() + 391.97 )
			Xnew = np.add(Xi , np.multiply(np.subtract(Xbest,Xworst),random.random()*(1/golden)) )
			Xnew = toBinaryX(Xnew,dimension,population[i],trainX, testX, trainy, testy)
			#print('Xnew: ', Xnew)
			FITnew = fitness(Xnew, trainX, testX, trainy, testy)
			#print('FITnew: ', FITnew)
			# if FITnew < fitList[i]:
			fitList[i] = FITnew
			population[i] = Xnew.copy()

			if fitList[i]< BESTACC:
				BESTACC = fitList[i]
				BESTANS = population[i].copy()
			######################################################
			####################  FITNESS  #######################
			######################################################
			cols_ = np.flatnonzero(population[i])
			if np.shape(cols_)[0]!=0:
				clf = KNeighborsClassifier(n_neighbors=5)
				train_data_pop = trainX[:,cols_]
				test_data_pop = testX[:,cols_]
				clf.fit(train_data_pop,trainy)
				acc = clf.score(test_data_pop,testy)
				print('population: ', population[i])
				print('acc: ', acc)
				if len(population_list)<=popSize:
					population_list.append(population[i]) 
					accuracies.append(acc)
				else :
					if acc > min(accuracies):
						minIdx = np.argmin(accuracies)
						accuracies[minIdx] = acc
						population_list[minIdx] = population[i]



		# pyplot.plot(x_axis,y_axis)
		# pyplot.show()
		# bestInx = np.argmin(fitList)
		# fitBest = min(fitList)
		# Xbest = population[bestInx].copy()
	cols = np.flatnonzero(BESTANS)
	val = 1
	if np.shape(cols)[0]==0:
		return Xbest
	clf = KNeighborsClassifier(n_neighbors=5)
	train_data = trainX[:,cols]
	test_data = testX[:,cols]
	clf.fit(train_data,trainy)
	val = clf.score(test_data,testy)
	return BESTANS,val, accuracies, population_list




#==================================================================
golden = (1 + 5 ** 0.5) / 2
popSize = 10
maxIter = 10
omega = 1


##### KFold Validations ########

from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
Fold = 5

kfold = KFold(Fold, True, 1)

f = 0

for train_index, test_index in kfold.split(data):
		#for dataset in datasetList:
		accuList = []
		featList = []
		for count in range(3):
			print("\nFold: {}, Iteration : {}".format(f+1,count+1))
			#if (dataset == "WaveformEW" or dataset == "KrVsKpEW") and count>2:
			#	break
			print(count)
			answer,testAcc,accuracies, population_list = goldenratiomethod(train_index, test_index,popSize,maxIter)
			#print('accuracies', accuracies)
			print(testAcc,answer.sum())
			accuList.append(testAcc)
			featList.append(answer.sum())
		inx = np.argmax(accuList)
		best_accuracy = accuList[inx]
		best_no_features = featList[inx]
		print("best:",accuList[inx],featList[inx])
		f += 1
'''
		with open("result_GRx.csv","a") as f:
			print(dataset,"%.2f" % (100*best_accuracy),best_no_features,file=f)
'''

