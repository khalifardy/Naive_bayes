#%%
import pandas as pd
import numpy as np
import math
# %%
def read_excel(path, sheet_target):
    data = pd.read_excel(path, sheet_name=sheet_target)
    return data
 

# %%
datatrain = read_excel("newTrain.xlsx",'train')
datatest = read_excel("newTest.xlsx",'test')
dataTruth = read_excel("newTestGroundTruth.xlsx","truth")

# %%
datatest
# %%
datatrain
# %%
dataTruth


# %%
datatrain = datatrain.drop("Unnamed: 0", axis=1)
datatest = datatest.drop("Unnamed: 0", axis=1)
dataTruth = dataTruth.drop("Unnamed: 0",axis=1)
# %%
datatrain
# %%
def normalizationMinMax(data, kolomtarget):
    for kolom in kolomtarget:
        data[kolom] = (data[kolom]-data[kolom].min())/(data[kolom].max()-data[kolom].min())
    return data
# %%
def standardization(data, kolomtarget):
    for kolom in kolomtarget:
        data[kolom] = (data[kolom]-data[kolom].mean())/(data[kolom].std())
    return data

# %%
def calc_probability(mean,std,x):
    exponent = math.exp(-((x-mean)**2/(2* std**2)))
    return 1/(math.sqrt(2*math.pi)*std)*exponent
# %%
def splitTruth(dataTrain, kolomtarget):
    truthData = []
    for truth in dataTrain[kolomtarget].unique():
        truthData.append(dataTrain.where(dataTrain[kolomtarget]==truth).dropna())
    return truthData
# %%
def find_mean(yesData,noData, kolomtarget):
    yesMean= dict()
    noMean = dict()

    for kolom in kolomtarget:
        yesMean[kolom] = yesData[kolom].mean()
        noMean[kolom] = noData[kolom].mean()
    
    return yesMean, noMean


# %%
def find_std(yesData, noData, kolomTarget):
    yesStd = dict()
    noStd = dict()
    for kolom in kolomTarget:
        yesStd[kolom] = yesData[kolom].std()
        noStd[kolom] = noData[kolom].std()
    
    return yesStd, noStd

# %%
def doPrediction(yesMean, yesStd, noMean, noStd, target, kolomtarget, truthColumn):
    result = []
    for i in range(len(target)):
        yesResult = 1
        noResult = 1

        for kolom in kolomtarget:
            yesResult *= calc_probability(yesMean[kolom], yesStd[kolom], target[kolom].iloc[i])
            noResult *= calc_probability(noMean[kolom], noStd[kolom], target[kolom].iloc[i])

        result.append({
            'ID':target['id'].iloc[i], 'Yes Probability':"{}".format(yesResult), "No Probability": "{}".format(noResult), 'Prediction Result':int(yesResult>noResult),
            'Ground Truth': target[truthColumn].iloc[i]
        })

    
    return result


# %%
def folding(dataset, trainingPercentage, location, shuffle=bool):
    lengthTraining = int(len(dataset)*trainingPercentage/100)

    if (shuffle):
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    train = []
    validation = []

    if (location == "left"):
        train, validation = dataset.iloc[:lengthTraining].reset_index(drop=True), dataset.iloc[lengthTraining].reset_index(drop=True)
    elif (location == "right"):
        validation,train = dataset.iloc[:abs(lengthTraining-len(dataset))].reset_index(drop=True), dataset.iloc[abs(lengthTraining-len(dataset)):].reset_index(drop=True)
    elif (location == 'meddle'):
        train = dataset.iloc[int(abs(lengthTraining-len(dataset))/2):len(dataset)-int(abs(lengthTraining-len(dataset))/2)]
        validation = pd.concat([dataset.iloc[:int(abs(lengthTraining-len(dataset))/2)],dataset.iloc[len(dataset)-int(abs(lengthTraining-len(dataset))/2):]])
    
    return train,validation

# %%

def conffusionMatrix(result):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    x = True

    for i in result:
        if i["Ground Truth"]=="?":
            x = False
            break

        elif i['Prediction Result'] == 1 and i['Prediction Result'] == i["Ground Truth"]:
            TP += 1
        
        elif i['Prediction Result'] == 0 and i['Prediction Result'] == i["Ground Truth"]:
            TN += 1
        
        elif i['Prediction Result'] == 1 and i['Prediction Result'] != i["Ground Truth"]:
            FP += 1
        elif i['Prediction Result'] == 0 and i['Prediction Result'] != i["Ground Truth"]:
            FN += 1
        
    
    if x :
        print(f"\nTP:{TP} FN:{FN}\nTN:{TN} FN:{FN}")
        print(f"accuracy :  {((TP+TN)/(TP+TN+FP+FN))*100}%")
        print(f"Precission: {((TP)/(TP+FP))*100}%")
        print(f"recall: {((TP)/(TP+FN))*100}%")
    else:
        print("tidak bisa memproses data")
# %%
#EXCERSISE
dataku = pd.read_csv("Data.csv")
# %%
id = [i+1 for i in range(len(dataku))]
# %%
dataku["id"] = id
# %%
dataku.info()
# %%
train,validation = folding(dataku.copy(),70,"right",shuffle=True)
# %%
idTrain = train.index.stop
semuaData = pd.concat([train,validation])
semuaData = standardization(semuaData,["glucose","bloodpressure"])
stdTrain,stdTest = semuaData.iloc[:idTrain].drop("id",axis=1), semuaData.iloc[idTrain:].drop('id', axis=1)

# %%
yesData, noData = splitTruth(train, "diabetes")
# %%
yesMean, noMean = find_mean(yesData,noData,["glucose","bloodpressure"])
# %%
print(f"Mean Result \n1 : {yesMean}\n0:{noMean}")
# %%
yesStd,noStd = find_std(yesData,noData,["glucose","bloodpressure"])
# %%
print(f"STD Result \n1 : {yesStd}\n0:{noStd}")
# %%
target = validation

result = doPrediction(yesMean,yesStd,noMean,noStd,target,["glucose","bloodpressure"],"diabetes")
 
# %%
for p in result:
    print(p)

conffusionMatrix(result)
# %%
