library(modopt.matlab)
library(dplyr)
library(kernlab)
library(e1071)
setwd('C:/Users/liutiantian/Desktop/KM4CQG/subFun/realMissing/pkgFun')
source('tuningParamSet1.R')
source('misCovLearner.R')
source('myCVMeasure.R')
source('MImodels.R')
source('pathes.R')
hepaDat=read.csv('hepa.csv')[,-1]
hepaDatCC=hepaDat[(hepaDat$R1==1)*(hepaDat$R2==1)==1,]
hepaFun=function(seed){
  set.seed(seed)
  testLabel=sample(c(1:80),30,replace = F)
  misCovDat=rbind(hepaDatCC[setdiff(c(1:80),testLabel),],
                  hepaDat[(hepaDat$R1==1)*(hepaDat$R2==1)!=1,])
  dataTst=hepaDatCC[testLabel,]
  
  px=4 # dimension of fully observed covariates
  p=6  # dimension of covariates
  
## Step 1: Make task
regr.task=makeRegrTask(id="misCov",data=misCovDat,target = "Y")

## Step 2: Make learner for corss validation
# 5 flods for cv
ctrl = makeTuneControlGrid()
rdesc = makeResampleDesc("CV", iters = 5L)

## CC method
discrete_psCC=tuningParamSet1(p-2,"CC","RBF",psPath0,impPath0,1)
resCC= tuneParams("regr.misCovSVMreal", task = regr.task, resampling = rdesc,
                  measures=my.cv.measure, par.set = discrete_psCC, control = ctrl)
par.valsCC= resCC$x # get the best parameter of lambda and sigma
par.valsCC[["testPurpose"]]= "testing"
lrnCC= setHyperPars(makeLearner("regr.misCovSVMreal"), par.vals = par.valsCC) # Make the learner
# Step 3: Train the learner
trainAlgoCC= train(lrnCC,regr.task)
# Step 4: Test the learner using the testing data set
testCC=predict(trainAlgoCC,newdata=dataTst)


## WCC method (if propensity score is correctly specified, psPath1; misspecified, psPath0).
discrete_psWCC_CM=tuningParamSet1(4,"WCC","RBF",psPath1,impPath0,1)
resWCC_CM= tuneParams("regr.misCovSVMreal", task = regr.task, resampling = rdesc,
                      measures=my.cv.measure, par.set = discrete_psWCC_CM, 
                      control = ctrl)
par.valsWCC_CM= resWCC_CM$x # get the best parameter of lambda and sigma
par.valsWCC_CM[["testPurpose"]]= "testing"
lrnWCC_CM= setHyperPars(makeLearner("regr.misCovSVMreal"), par.vals =par.valsWCC_CM) # Make learner
trainAlgoWCC_CM= train(lrnWCC_CM,regr.task) # training
testWCC_CM=predict(trainAlgoWCC_CM,newdata=dataTst) # testing

## misspecified the propensity score model
discrete_psWCC_MM=tuningParamSet1(4,"WCC","RBF",psPath0,impPath0,1)
resWCC_MM= tuneParams("regr.misCovSVMreal", task = regr.task, resampling = rdesc,
                      measures=my.cv.measure, par.set = discrete_psWCC_MM, control = ctrl)
par.valsWCC_MM= resWCC_MM$x
par.valsWCC_MM[["testPurpose"]]= "testing"
lrnWCC_MM= setHyperPars(makeLearner("regr.misCovSVMreal"), par.vals =par.valsWCC_MM) # Make learner
trainAlgoWCC_MM= train(lrnWCC_MM,regr.task) # training
testWCC_MM=predict(trainAlgoWCC_MM,newdata=dataTst) # testing

## both correctly specified
discrete_psDR_CMCIMP=tuningParamSet1(4,"DR","RBF",psPath1,impPath1,2) # PSmodel=1, Regmodel=1
resDR_CMCIMP= tuneParams("regr.misCovSVMreal", task = regr.task, resampling = rdesc,
                         measures=my.cv.measure, par.set = discrete_psDR_CMCIMP, control = ctrl)
par.valsDR_CMCIMP= resDR_CMCIMP$x # get the best parameter of lambda and sigma
par.valsDR_CMCIMP[["testPurpose"]]= "testing"
lrnDR_CMCIMP= setHyperPars(makeLearner("regr.misCovSVMreal"), par.vals = par.valsDR_CMCIMP) # Make learner
trainAlgoDR_CMCIMP= train(lrnDR_CMCIMP,regr.task) # trainging
testDR_CMCIMP=predict(trainAlgoDR_CMCIMP,newdata=dataTst) # testing


## propensity score correctly specified, imputation misspecified
discrete_psDR_CMMIMP=tuningParamSet1(4,"DR","RBF",psPath1,impPath0,5) # PSmodel=1, Regmodel=0
resDR_CMMIMP= tuneParams("regr.misCovSVMreal", task = regr.task, resampling = rdesc,
                         measures=my.cv.measure, par.set = discrete_psDR_CMMIMP, control = ctrl)
par.valsDR_CMMIMP= resDR_CMMIMP$x # get the best parameter of lambda and sigma
par.valsDR_CMMIMP[["testPurpose"]]= "testing"
lrnDR_CMMIMP= setHyperPars(makeLearner("regr.misCovSVMreal"), par.vals = par.valsDR_CMMIMP) # Make learner
trainAlgoDR_CMMIMP= train(lrnDR_CMMIMP,regr.task) # trainging
testDR_CMMIMP=predict(trainAlgoDR_CMMIMP,newdata=dataTst) # testing

## propensity score misspecified, imputation correctly specified
discrete_psDR_MMCIMP=tuningParamSet1(4,"DR","RBF",psPath0,impPath1,5)
resDR_MMCIMP= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                         measures=my.cv.measure, par.set = discrete_psDR_MMCIMP, control = ctrl)
par.valsDR_MMCIMP= resDR_MMCIMP$x # get the best parameter of lambda and sigma
par.valsDR_MMCIMP[["testPurpose"]]= "testing"
lrnDR_MMCIMP= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals = par.valsDR_MMCIMP) # Make learner
trainAlgoDR_MMCIMP= train(lrnDR_MMCIMP,regr.task) # trainging
testDR_MMCIMP=predict(trainAlgoDR_MMCIMP,newdata=dataTst) # testing

## both misspecified
discrete_psDR_MMMIMP=tuningParamSet1(4,"DR","RBF",psPath0,impPath0,5)
resDR_MMMIMP= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                         measures=my.cv.measure, par.set = discrete_psDR_MMMIMP, control = ctrl)
par.valsDR_MMMIMP= resDR_MMMIMP$x # get the best parameter of lambda and sigma
par.valsDR_MMMIMP[["testPurpose"]]= "testing"
lrnDR_MMMIMP= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals = par.valsDR_MMMIMP) # Make learner
trainAlgoDR_MMMIMP= train(lrnDR_MMMIMP,regr.task) # trainging
testDR_MMMIMP=predict(trainAlgoDR_MMMIMP,newdata=dataTst) # testing

library(e1071)
# CC
misCovDat_CC=misCovDat[misCovDat$R2==1,]
CC_S<- best.svm(
  x = misCovDat_CC[,1:p]
  , y = factor(misCovDat_CC$Y)
  , type = "C-classification"
  , kernel = "radial"
  , cost =p^(-6:6)# 1/(2*nTrain*10^(-3:1)) #cost = 2^(-6:6)
)
CC_S=predict(CC_S, dataTst[,1:p])
CC_S=2*(as.numeric(CC_S)-1)-1

library(DMwR)
misCovDat$x18[misCovDat$R1==0]=NA
misCovDat$x19[misCovDat$R2==0]=NA
misCovDat_meanIMP=misCovDat
misCovDat_meanIMP$x18[is.na(misCovDat_meanIMP$x18)]=mean(misCovDat_meanIMP$x18[misCovDat_meanIMP$R1==1])
misCovDat_meanIMP$x19[is.na(misCovDat_meanIMP$x19)]=mean(misCovDat_meanIMP$x19[misCovDat_meanIMP$R2==1])
trainAlgoIMP_mean <- best.svm(
  x = misCovDat_meanIMP[,1:p]
  , y = factor(misCovDat_meanIMP$Y)
  , type = "C-classification"
  , kernel = "radial"
  , cost =p^(-6:6)#1/(2*nTrain*10^(-3:1)) # 2^(-6:6)
)

testIMP_mean=predict(trainAlgoIMP_mean, dataTst[,1:p])
testIMP_mean=2*(as.numeric(testIMP_mean)-1)-1

misCovDat_KnnIMP=knnImputation(misCovDat,k=10)
trainAlgoIMP_KNN <- best.svm(
  x = misCovDat_KnnIMP[,1:p]
  , y = factor(misCovDat_KnnIMP$Y)
  , type = "C-classification"
  , kernel = "radial"
  , cost =p^(-6:6)#1/(2*nTrain*10^(-3:1)) # 2^(-6:6)
)
testIMP_KNN=predict(trainAlgoIMP_KNN, dataTst[,1:p])
testIMP_KNN=2*(as.numeric(testIMP_KNN)-1)-1


## multiple imputation
pTotal=dim(misCovDat)[2] # dimension of (X,V,R,Y1,Y)
misCovDat=as.matrix(misCovDat)
names(misCovDat)=NULL
X=misCovDat[,1:px] # the fully observed covariates
V=misCovDat[,(px+1):(pTotal-4)] # the popential missing covariates
Z=cbind(X,V) # the covariates
R1=misCovDat[,(pTotal-3)]
R2=misCovDat[,(pTotal-2)]   #  missing indicator
Y=misCovDat[,pTotal] # the response (-1 and 1)
n=nrow(misCovDat) # sample size of the traing data
Index=((R1==1)*(R2==1))
YCC=Y[Index==1] # Y corresponding complete cases
ZCC=Z[Index==1,] # covariates corresponding complete cases
XCC=Z[Index==1,1:px] # Y corresponding complete cases
VCC=Z[Index==1,(px+1):p] # covariates corresponding complete cases
YIC=Y[R1==0] # Y corresponding incomplete cases
XIC=Z[R1==0,1:px] # Y corresponding incomplete cases
XPC=Z[((R1==1)*(R2==0))==1,1:(px+1)]
YPC=Y[((R1==1)*(R2==0))==1]
dataIC=cbind(XIC,YIC) # observed incomplete data
dataPC=cbind(XPC,YPC) 
dataCC=cbind(XCC,VCC,YCC) # complete data
B=5 ## imputation times

## imputation under model1
IMP_MICIMP1=modelImp_MIcorrect(XCC,YCC,VCC)
IMP_MICIMP2=modelImp_MIcorrect(cbind(XCC,VCC[,1]),YCC,VCC[,2])
resIMP_MICIMPVec=rep(0,B)
for(i in 1:B){
  imputeData1=t(apply(dataIC,1,IMP_MICIMP1))
  imputeData2=t(apply(dataPC,1,IMP_MICIMP2))
  fullData=rbind(dataCC,imputeData1,imputeData2)
  fullData[fullData[,5]<0.5,5]=0.5
  fullData[fullData[,6]<0.5,6]=0.5
  fullData=as.data.frame(fullData)
  cloName=c("x1","x14","x15","x16","x18","x19","Y")
  names(fullData) <- cloName
  trainAlgoIMP_MICIMP<- best.svm(
    x = fullData[,1:p]
    , y = factor(fullData$Y)
    , type = "C-classification"
    , kernel = "radial"
    , cost =p^(-6:6)#1/(2*nTrain*10^(-3:1)) # 2^(-6:6)
  )
  testIMP_MICIMP=predict(trainAlgoIMP_MICIMP, dataTst[,1:p])
  
  testIMP_MICIMP=2*(as.numeric(testIMP_MICIMP)-1)-1
  resIMP_MICIMP=mean((testIMP_MICIMP-dataTst$Y)^2)
  resIMP_MICIMPVec[i]=resIMP_MICIMP}

## imputation under model0
IMP_MIMIMP1=modelImp_MImiss(XCC,YCC,VCC)
IMP_MIMIMP2=modelImp_MImiss(cbind(XCC,VCC[,1]),YCC,VCC[,2])
resIMP_MIMIMPVec=rep(0,B)
for(i in 1:B){
  imputeData1=t(apply(dataIC,1,IMP_MIMIMP1))
  imputeData2=t(apply(dataPC,1,IMP_MIMIMP2))
  fullData=rbind(dataCC,imputeData1,imputeData2)
  fullData[fullData[,5]<0.5,5]=0.5
  fullData[fullData[,6]<0.5,6]=0.5
  fullData=as.data.frame(fullData)
  cloName=c("x1","x14","x15","x16","x18","x19","Y")
  names(fullData) <- cloName
  trainAlgoIMP_MIMIMP <- best.svm(
    x = fullData[,1:p]
    , y = factor(fullData$Y)
    , type = "C-classification"
    , kernel = "radial"
    , cost =p^(-6:6)#1/(2*nTrain*10^(-3:1))
  )
  testIMP_MIMIMP=predict(trainAlgoIMP_MIMIMP, dataTst[,1:p])
  testIMP_MIMIMP=2*(as.numeric(testIMP_MIMIMP)-1)-1
  resIMP_MIMIMP=mean((testIMP_MIMIMP-dataTst$Y)^2)
  resIMP_MIMIMPVec[i]=resIMP_MIMIMP}

YTst=dataTst$Y
rate_pos=mean(misCovDat[,10]==1)

YCC=testCC$data[,2]
YCC[YCC==0]=2*(rbinom(sum(YCC==0),1,rate_pos))-1
mseCC=mean((YTst-YCC)^2)

YWCC_CM=testWCC_CM$data[,2]
YWCC_CM[YWCC_CM==0]=2*(rbinom(sum(YWCC_CM==0),1,rate_pos))-1
mseWCC_CM=mean((YTst-YWCC_CM)^2)

YWCC_MM=testWCC_MM$data[,2]
YWCC_MM[YWCC_MM==0]=2*(rbinom(sum(YWCC_MM==0),1,rate_pos))-1
mseWCC_MM=mean((YTst-YWCC_MM[,2])^2)

YDR_CMCIMP=testDR_CMCIMP$data[,2]
YDR_CMCIMP[YDR_CMCIMP==0]=2*(rbinom(sum(YDR_CMCIMP==0),1,rate_pos))-1
mseDR_CMCIMP=mean((YTst-YDR_CMCIMP)^2)

YDR_CMMIMP=testDR_CMMIMP$data[,2]
YDR_CMMIMP[YDR_CMMIMP==0]=2*(rbinom(sum(YDR_CMMIMP==0),1,rate_pos))-1
mseDR_CMCIMP=mean((YTst-YDR_CMMIMP)^2)


YDR_MMCIMP=testDR_MMCIMP$data[,2]
YDR_MMCIMP[YDR_MMCIMP==0]=2*(rbinom(sum(YDR_MMCIMP==0),1,rate_pos))-1
mseDR_MMCIMP=mean((YTst-YDR_MMCIMP$data[,2])^2)

YDR_MMMIMP=testDR_MMMIMP$data[,2]
YDR_MMMIMP[YDR_MMMIMP==0]=2*(rbinom(sum(YDR_MMMIMP==0),1,rate_pos))-1
mseDR_MMMIMP=mean((YTst-YDR_MMMIMP$data[,2])^2)


mseIMP_mean=mean((testIMP_mean-YTst)^2)
mseIMP_KNN=mean((testIMP_KNN-YTst)^2)
mseIMP_MICIMP=mean(resIMP_MICIMPVec)
mseIMP_MIMIMP=mean(resIMP_MIMIMPVec)

mseCC_S=mean((YTst-CC_S)^2)

myDF=data.frame(seed = seed,mseCC=mseCC,
                mseWCC_CM=mseWCC_CM,mseWCC_MM=mseWCC_MM,
                mseDR_CMCIMP=mseDR_CMCIMP,mseDR_MMCIMP=mseDR_MMCIMP,
                mseDR_CMMIMP=mseDR_CMMIMP,mseDR_MMMIMP=mseDR_MMMIMP,
                mseIMP_mean=mseIMP_mean,mseIMP_KNN=mseIMP_KNN,
                mseIMP_MICIMP=mseIMP_MICIMP,mseIMP_MIMIMP=mseIMP_MIMIMP,
                mseCC_S=mseCC_S)

}

