verticalFun=function(seed){ 
  setwd('C:/Users/LiuTT/Desktop/KM4CQGfiles/subFun/vertical')
  vData=read.csv("verticalData.csv")
  set.seed(seed)
  
  ## training sample of size 200
  trainSampleSize=200
  trainLable=sample(c(1:310),trainSampleSize,replace = F)
  trainData=vData[trainLable,]
  
  ## testing sample of size 200
  testLabel=setdiff(c(1:310),trainLable)
  testData=vData[testLabel,]
  
  x1=trainData$x1
  x2=trainData$x2
  x3=trainData$x3
  x4=trainData$x4
  x5=trainData$x5
  x6=trainData$x6
  px=4 # dimension of fully observed covariates
  p=6  # dimension of covariates
  
  alpha=2#3#1#2.5#2#1#2#3#2#1#-2#-2/3#1#-1#-1/5#-1/2##-1/15#-1/2#-1/4#-1/4#-1/4#-3#-1/4#-1/2#-4#-2
  beta=-11#-16#-14#-11.5#-13#-12#-11#-10#-9#-8#1/2#-2#2#-4#-5#-8#-1#-10#-1/2#-9#-8#-6#4#-5#-4#-2#5#-1#1.5#3
  stdx1=(x1-mean(x1))/sd(x1)
  stdx2=(x2-mean(x2))/sd(x2)
  PScov=(stdx1+stdx2)/2
  probR=exp(alpha+beta*(PScov*trainData$Y))/(1+exp(alpha+beta*(PScov*trainData$Y)))
  R=rbinom(trainSampleSize,1,probR)
  mean(R==1)
  mean(trainData$Y==1)
  mean(trainData$Y[R==1]==1)
  
  misCovDat=as.data.frame(cbind(x1,x2,x3,x6,x4,x5,R,trainData$Y,trainData$Y1))
  names(misCovDat)=c("x1","x2","x3","x4","x5","x6","R","Y","Y1")
  
  ## rename as x1 to x6 for testing dataset
  Tx1=testData$x1
  Tx2=testData$x2
  Tx3=testData$x3
  Tx4=testData$x4
  Tx5=testData$x5
  Tx6=testData$x6
  TR=rep(1,length(Tx1))
  
  ## arrange the variables as (X,V,R£¬Y£¬Y1)
  dataTst=as.data.frame(cbind(Tx1,Tx2,Tx3,Tx6,Tx4,Tx5,TR,testData$Y,testData$Y1))
  names(dataTst)=c("x1","x2","x3","x4","x5","x6","R","Y","Y1")
  
  source('misCovLearner.R')
  
  ## Tuning paramers set for learner
  source('tuningParamSet1.R')
  
  ## Measure function
  source('myCVMeasure.R')
  
  ## pathes for prpensity score model fucntions and imputation model functions
  source('pathes.R')
  
  ## models for multiple imputation
  source('MImodels.R')
  
  ## Step 1: Make task
  regr.task=makeRegrTask(id="misCov",data=misCovDat,target = "Y")
  
  ## Step 2: Make learner for corss validation
  # 5 flods for cv
  ctrl = makeTuneControlGrid()
  rdesc = makeResampleDesc("CV", iters = 5L)
  
  ## CC method
  discrete_psCC=tuningParamSet1(px,"CC","RBF",psPath0,impPath0,1)
  resCC= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                    measures=my.cv.measure, par.set = discrete_psCC, control = ctrl)
  
  par.valsCC= resCC$x # get the best parameter of lambda and sigma
  par.valsCC[["testPurpose"]]= "testing"
  lrnCC= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals = par.valsCC) # Make the learner
  # Step 3: Train the learner
  trainAlgoCC= train(lrnCC,regr.task)
  # Step 4: Test the learner using the testing data set
  testCC=predict(trainAlgoCC,newdata=dataTst)
  
  
  ## WCC method (if propensity score is correctly specified, psPath1; misspecified, psPath0).
  discrete_psWCC_CM=tuningParamSet1(px,"WCC","RBF",psPath1,impPath0,1)
  resWCC_CM= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                        measures=my.cv.measure, par.set = discrete_psWCC_CM, 
                        control = ctrl)
  par.valsWCC_CM= resWCC_CM$x # get the best parameter of lambda and sigma
  par.valsWCC_CM[["testPurpose"]]= "testing"
  lrnWCC_CM= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals =par.valsWCC_CM) # Make learner
  trainAlgoWCC_CM= train(lrnWCC_CM,regr.task) # training
  testWCC_CM=predict(trainAlgoWCC_CM,newdata=dataTst) # testing
  
  ## misspecified the propensity score model
  discrete_psWCC_MM=tuningParamSet1(px,"WCC","RBF",psPath0,impPath0,1)
  resWCC_MM= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                        measures=my.cv.measure, par.set = discrete_psWCC_MM, control = ctrl)
  par.valsWCC_MM= resWCC_MM$x
  par.valsWCC_MM[["testPurpose"]]= "testing"
  lrnWCC_MM= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals =par.valsWCC_MM) # Make learner
  trainAlgoWCC_MM= train(lrnWCC_MM,regr.task) # training
  testWCC_MM=predict(trainAlgoWCC_MM,newdata=dataTst) # testing
  
  ## both correctly specified
  discrete_psDR_CMCIMP=tuningParamSet1(px,"DR","RBF",psPath1,impPath1,5) # PSmodel=1, Regmodel=1
  resDR_CMCIMP= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                           measures=my.cv.measure, par.set = discrete_psDR_CMCIMP, control = ctrl)
  par.valsDR_CMCIMP= resDR_CMCIMP$x # get the best parameter of lambda and sigma
  par.valsDR_CMCIMP[["testPurpose"]]= "testing"
  lrnDR_CMCIMP= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals = par.valsDR_CMCIMP) # Make learner
  trainAlgoDR_CMCIMP= train(lrnDR_CMCIMP,regr.task) # trainging
  testDR_CMCIMP=predict(trainAlgoDR_CMCIMP,newdata=dataTst) # testing


  ## propensity score correctly specified, imputation misspecified
  discrete_psDR_CMMIMP=tuningParamSet1(px,"DR","RBF",psPath1,impPath0,5) # PSmodel=1, Regmodel=0
  resDR_CMMIMP= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                           measures=my.cv.measure, par.set = discrete_psDR_CMMIMP, control = ctrl)
  par.valsDR_CMMIMP= resDR_CMMIMP$x # get the best parameter of lambda and sigma
  par.valsDR_CMMIMP[["testPurpose"]]= "testing"
  lrnDR_CMMIMP= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals = par.valsDR_CMMIMP) # Make learner
  trainAlgoDR_CMMIMP= train(lrnDR_CMMIMP,regr.task) # trainging
  testDR_CMMIMP=predict(trainAlgoDR_CMMIMP,newdata=dataTst) # testing

  ## propensity score misspecified, imputation correctly specified
  discrete_psDR_MMCIMP=tuningParamSet1(px,"DR","RBF",psPath0,impPath1,5)
  resDR_MMCIMP= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                           measures=my.cv.measure, par.set = discrete_psDR_MMCIMP, control = ctrl)
  par.valsDR_MMCIMP= resDR_MMCIMP$x # get the best parameter of lambda and sigma
  par.valsDR_MMCIMP[["testPurpose"]]= "testing"
  lrnDR_MMCIMP= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals = par.valsDR_MMCIMP) # Make learner
  trainAlgoDR_MMCIMP= train(lrnDR_MMCIMP,regr.task) # trainging
  testDR_MMCIMP=predict(trainAlgoDR_MMCIMP,newdata=dataTst) # testing

  ## both misspecified
  discrete_psDR_MMMIMP=tuningParamSet1(px,"DR","RBF",psPath0,impPath0,5)
  resDR_MMMIMP= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                           measures=my.cv.measure, par.set = discrete_psDR_MMMIMP, control = ctrl)
  par.valsDR_MMMIMP= resDR_MMMIMP$x # get the best parameter of lambda and sigma
  par.valsDR_MMMIMP[["testPurpose"]]= "testing"
  lrnDR_MMMIMP= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals = par.valsDR_MMMIMP) # Make learner
  trainAlgoDR_MMMIMP= train(lrnDR_MMMIMP,regr.task) # trainging
  testDR_MMMIMP=predict(trainAlgoDR_MMMIMP,newdata=dataTst) # testing

  
  library(e1071)
  # CC
  misCovDat_CC=misCovDat[misCovDat$R==1,]
  CC_S<- best.svm(
    x = misCovDat_CC[,1:p]
    , y = factor(misCovDat_CC$Y)
    , type = "C-classification"
    , kernel = "radial"
    , cost =p^(-6:6)# 1/(2*nTrain*10^(-3:1)) #cost = 2^(-6:6)
  )
  CC_S=predict(CC_S, dataTst[,1:p])
  CC_S=2*(as.numeric(CC_S)-1)-1
  
  SVMfullData<- best.svm(
    x = misCovDat[,1:p]
    , y = factor(misCovDat$Y)
    , type = "C-classification"
    , kernel = "radial"
    , cost =p^(-6:6)#1/(2*nTrain*10^(-3:1)) #2^(-5:5)
  )
  SVMfullData=predict(SVMfullData, dataTst[,1:p])
  SVMfullData=2*(as.numeric(SVMfullData)-1)-1
  
  
  library(DMwR)
  misCovDat$x5[misCovDat$R==0]=NA
  misCovDat$x6[misCovDat$R==0]=NA
  misCovDat_meanIMP=misCovDat
  misCovDat_meanIMP$x5[is.na(misCovDat_meanIMP$x5)]=mean(misCovDat_meanIMP$x5[misCovDat_meanIMP$R==1])
  misCovDat_meanIMP$x6[is.na(misCovDat_meanIMP$x6)]=mean(misCovDat_meanIMP$x6[misCovDat_meanIMP$R==1])
  trainAlgoIMP_mean <- best.svm(
    x = misCovDat_meanIMP[,1:p]
    , y = factor(misCovDat_meanIMP$Y)
    , type = "C-classification"
    , kernel = "radial"
    , cost =p^(-6:6)#1/(2*nTrain*10^(-3:1)) # 2^(-6:6)
  )
  
  testIMP_mean=predict(trainAlgoIMP_mean, dataTst[,1:p])
  testIMP_mean=2*(as.numeric(testIMP_mean)-1)-1
  
  ## knn imputation
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
  V=misCovDat[,(px+1):(pTotal-3)] # the popential missing covariates
  Z=cbind(X,V) # the covariates
  R=misCovDat[,(pTotal-2)]   #  missing indicator
  Y=misCovDat[,pTotal] # the response (-1 and 1)
  n=nrow(misCovDat) # sample size of the traing data
  YCC=Y[R==1] # Y corresponding complete cases
  ZCC=Z[R==1,] # covariates corresponding complete cases
  XCC=Z[R==1,1:px] # Y corresponding complete cases
  VCC=Z[R==1,(px+1):p] # covariates corresponding complete cases
  YIC=Y[R==0] # Y corresponding incomplete cases
  ZIC=Z[R==0,] # covariates corresponding incomplete cases
  XIC=Z[R==0,1:px] # Y corresponding incomplete cases
  dataIC=cbind(XIC,Y[R==0]) # observed incomplete data
  dataCC=cbind(XCC,VCC,YCC) # complete data
  B=5 ## imputation times
  
  ## imputation under model1
  IMP_MICIMP=modelImp_MIcorrect(XCC,YCC,VCC)
  resIMP_MICIMPVec=rep(0,B)
  for(i in 1:B){
    imputeData=t(apply(dataIC,1,IMP_MICIMP))
    fullData=rbind(dataCC,imputeData)
    fullData=as.data.frame(fullData)
    if(p==2){cloName=c("x2","x1","Y")}else{cloName=c(paste0("x",2:p),"x1","Y")}
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
  IMP_MIMIMP=modelImp_MImiss(XCC,YCC,VCC)
  resIMP_MIMIMPVec=rep(0,B)
  for(i in 1:B){
    imputeData=t(apply(dataIC,1,IMP_MIMIMP))
    fullData=rbind(dataCC,imputeData)
    fullData=as.data.frame(fullData)
    if(p==2){cloName=c("x2","x1","Y")}else{cloName=c(paste0("x",2:p),"x1","Y")}
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
  mseCC=mean((YTst-testCC$data[,2])^2)
  mseWCC_CM=mean((YTst-testWCC_CM$data[,2])^2)
  mseWCC_MM=mean((YTst-testWCC_MM$data[,2])^2)
  
  
  mseDR_CMCIMP=mean((YTst-testDR_CMCIMP$data[,2])^2)
  mseDR_CMMIMP=mean((YTst-testDR_CMMIMP$data[,2])^2)
  mseDR_MMCIMP=mean((YTst-testDR_MMCIMP$data[,2])^2)
  mseDR_MMMIMP=mean((YTst-testDR_MMMIMP$data[,2])^2)
  
  
  mseIMP_mean=mean((testIMP_mean-YTst)^2)
  mseIMP_KNN=mean((testIMP_KNN-YTst)^2)
  mseIMP_MICIMP=mean(resIMP_MICIMPVec)
  mseIMP_MIMIMP=mean(resIMP_MIMIMPVec)
  
  mseCC_S=mean((YTst-CC_S)^2)
  mseSVMfullData=mean((YTst-SVMfullData)^2)
  
  
  myDF=data.frame(seed = seed,mseCC=mseCC,
                  mseWCC_CM=mseWCC_CM,mseWCC_MM=mseWCC_MM,
                  mseDR_CMCIMP=mseDR_CMCIMP,mseDR_MMCIMP=mseDR_MMCIMP,
                  mseDR_CMMIMP=mseDR_CMMIMP,mseDR_MMMIMP=mseDR_MMMIMP,
                  mseIMP_mean=mseIMP_mean,mseIMP_KNN=mseIMP_KNN,
                  mseIMP_MICIMP=mseIMP_MICIMP,mseIMP_MIMIMP=mseIMP_MIMIMP,
                  mseCC_S=mseCC_S,mseSVMfullData=mseSVMfullData)
  
}

dfCom=data.frame(matrix(0,150,14))
names(dfCom)=c('seed','CC','WCC1','WCC2','DR1','DR2','DR3','DR4','IMP_M','IMP_KNN','IMP_MI1','IMP_MI2',
            'CC2','Full')
for(i in c(101:110)){
  message(paste("at",i))
  dfCom[i,]=verticalFun(i)
}
write.csv(dfCom,'verticalResAll_Com.csv')

