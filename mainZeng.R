setwd("/data/home/tiantian.liu/MisCov/subFun/setting2")
p=10
pR=2
missing=1
alpha=-1/2
beta=6
gamma=1/8
delta=6.3#6#5.5#7#7.3#5
ZengFun= function(seed,nlist = c(100,200)){
  library(DMwR)
  library(e1071)
  ## Linear
  source('misCovLearner.R')
  
  ## Tuning paramers set for learner
  source('tuningParamSet1.R')
  
  ## Measure function
  source('myCVMeasure.R')
  
  ## pathes for prpensity score model fucntions and imputation model functions
  source('pathes.R')
  
  ## models for multiple imputation
  source('MImodels.R')
  
  source('Zeng10Data.R')
  
  ## testing dataset
  
  nTst=10000
  misIndicTst=0
  set.seed(10000)
  dataTst=ZengData10(nTst,p,pR,misIndicTst,gamma,delta,alpha,beta)
  p=dim(dataTst)[2]-3
  px=p-2 # dimension of covariates
  
  nDifNum=length(nlist)
  mseCC=rep(0,nDifNum)
  mseCC=rep(0,nDifNum)
  
  mseWCC_CM=rep(0,nDifNum)
  mseWCC_MM=rep(0,nDifNum)
  
  mseDR_CMCIMP=rep(0,nDifNum)
  mseDR_CMMIMP=rep(0,nDifNum)
  mseDR_MMCIMP=rep(0,nDifNum)
  mseDR_MMMIMP=rep(0,nDifNum)
  
  mseIMP_mean=rep(0,nDifNum)
  mseIMP_KNN=rep(0,nDifNum)
  mseIMP_MICIMP=rep(0,nDifNum)
  mseIMP_MIMIMP=rep(0,nDifNum)
  
  mseCC_S=rep(0,nDifNum)
  mseSVMfullData=rep(0,nDifNum)
  
  for(ns in nlist){
    set.seed(seed+ns)
    
    nTrain=ns # Generate training data
    misCovDat=ZengData10(nTrain,p,pR,1,gamma,delta,alpha,beta)
    
    ## Step 1: Make task
    regr.task=makeRegrTask(id="misCov",data=misCovDat,target = "Y")
    
    ## Step 2: Make learner for corss validation
    # 5 flods for cv
    ctrl = makeTuneControlGrid()
    rdesc = makeResampleDesc("CV", iters = 5L)
    
    ## CC method
    #discrete_psCC=tuningParamSet(p-1,"CC")
    discrete_psCC=tuningParamSet1(px,"CC","linear",psPath0,impPath0,1)
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
    discrete_psWCC_CM=tuningParamSet1(px,"WCC","linear",psPath1,impPath0,1)
    resWCC_CM= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                          measures=my.cv.measure, par.set = discrete_psWCC_CM, 
                          control = ctrl)
    par.valsWCC_CM= resWCC_CM$x # get the best parameter of lambda and sigma
    par.valsWCC_CM[["testPurpose"]]= "testing"
    lrnWCC_CM= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals =par.valsWCC_CM) # Make learner
    trainAlgoWCC_CM= train(lrnWCC_CM,regr.task) # training
    testWCC_CM=predict(trainAlgoWCC_CM,newdata=dataTst) # testing
    
    ## misspecified the propensity score model
    discrete_psWCC_MM=tuningParamSet1(px,"WCC","linear",psPath0,impPath0,1)
    resWCC_MM= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                          measures=my.cv.measure, par.set = discrete_psWCC_MM, control = ctrl)
    par.valsWCC_MM= resWCC_MM$x
    par.valsWCC_MM[["testPurpose"]]= "testing"
    lrnWCC_MM= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals =par.valsWCC_MM) # Make learner
    trainAlgoWCC_MM= train(lrnWCC_MM,regr.task) # training
    testWCC_MM=predict(trainAlgoWCC_MM,newdata=dataTst) # testing
    
    # ## DR method (if propensity score is correctly specified, psPath1; misspecified, psPath0.
    # #            if imputation model is correctly specified, impPath1; misspecified, impPath0)
    #
    # both correctly specified
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
    
    # ## both misspecified
    discrete_psDR_MMMIMP=tuningParamSet1(px,"DR","RBF",psPath0,impPath0,5)
    resDR_MMMIMP= tuneParams("regr.misCovSVMcv", task = regr.task, resampling = rdesc,
                             measures=my.cv.measure, par.set = discrete_psDR_MMMIMP, control = ctrl)
    par.valsDR_MMMIMP= resDR_MMMIMP$x # get the best parameter of lambda and sigma
    par.valsDR_MMMIMP[["testPurpose"]]= "testing"
    lrnDR_MMMIMP= setHyperPars(makeLearner("regr.misCovSVMcv"), par.vals = par.valsDR_MMMIMP) # Make learner
    trainAlgoDR_MMMIMP= train(lrnDR_MMMIMP,regr.task) # trainging
    testDR_MMMIMP=predict(trainAlgoDR_MMMIMP,newdata=dataTst) # testing
    
    
    misCovDat_CC=misCovDat[misCovDat$R==1,]
    CC_S<- best.svm(
      x = misCovDat_CC[,1:p]
      , y = factor(misCovDat_CC$Y)
      , type = "C-classification"
      , kernel = "radial"
      , cost = p^(-6:6)
    )
    CC_S=predict(CC_S, dataTst[,1:p])
    CC_S=2*(as.numeric(CC_S)-1)-1
    
    SVMfullData<- best.svm(
      x = misCovDat[,1:p]
      , y = factor(misCovDat$Y)
      , type = "C-classification"
      , kernel = "radial"
      , cost = p^(-6:6)
    )
    SVMfullData=predict(SVMfullData, dataTst[,1:p])
    SVMfullData=2*(as.numeric(SVMfullData)-1)-1
    
    ## mean inputation
    misCovDat$x9[misCovDat$R==0]=NA
    misCovDat$x10[misCovDat$R==0]=NA
    misCovDat_meanIMP=misCovDat
    misCovDat_meanIMP$x9[is.na(misCovDat_meanIMP$x9)]=mean(misCovDat_meanIMP$x9[misCovDat_meanIMP$R==1])
    misCovDat_meanIMP$x10[is.na(misCovDat_meanIMP$x10)]=mean(misCovDat_meanIMP$x10[misCovDat_meanIMP$R==1])
    trainAlgoIMP_mean <- best.svm(
      x = misCovDat_meanIMP[,1:p]
      , y = factor(misCovDat_meanIMP$Y)
      , type = "C-classification"
      , kernel = "radial"
      , cost = p^(-6:6)
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
      , cost = p^(-6:6)
    )
    testIMP_KNN=predict(trainAlgoIMP_KNN, dataTst[,1:p])
    testIMP_KNN=2*(as.numeric(testIMP_KNN)-1)-1
    
    ## multiple imputation
    pTotal=dim(misCovDat)[2] # dimension of (X,V,R,Y1,Y)
    misCovDat=as.matrix(misCovDat)
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
    dataIC=cbind(XIC,YIC) # observed incomplete data
    dataCC=cbind(XCC,VCC,YCC) # complete data
    B=5 ## imputation times
    
    ## imputation is correctly specified
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
        , cost = p^(-6:6)
      )
      testIMP_MICIMP=predict(trainAlgoIMP_MICIMP, dataTst[,1:p])
      testIMP_MICIMP=2*(as.numeric(testIMP_MICIMP)-1)-1
      resIMP_MICIMP=mean((testIMP_MICIMP-dataTst$Y)^2)
      resIMP_MICIMPVec[i]=resIMP_MICIMP}
    
    
    ## imputation is misspecified
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
        , cost = p^(-6:6)#1/(2*nTrain*10^(-3:1)) 
      )
      testIMP_MIMIMP=predict(trainAlgoIMP_MIMIMP, dataTst[,1:p])
      testIMP_MIMIMP=2*(as.numeric(testIMP_MIMIMP)-1)-1
      resIMP_MIMIMP=mean((testIMP_MIMIMP-dataTst$Y)^2)
      resIMP_MIMIMPVec[i]=resIMP_MIMIMP}
    
    
    YTst=dataTst$Y
    ## mse
    mseCC[ns==nlist]=mean((YTst-testCC$data[,2])^2)
    mseWCC_CM[ns==nlist]=mean((YTst-testWCC_CM$data[,2])^2)
    mseWCC_MM[ns==nlist]=mean((YTst-testWCC_MM$data[,2])^2)
    
    mseDR_CMCIMP[ns==nlist]=mean((YTst-testDR_CMCIMP$data[,2])^2)
    mseDR_CMMIMP[ns==nlist]=mean((YTst-testDR_CMMIMP$data[,2])^2)
    mseDR_MMCIMP[ns==nlist]=mean((YTst-testDR_MMCIMP$data[,2])^2)
    mseDR_MMMIMP[ns==nlist]=mean((YTst-testDR_MMMIMP$data[,2])^2)
    
    mseIMP_mean[ns==nlist]=mean((testIMP_mean-YTst)^2)
    mseIMP_KNN[ns==nlist]=mean((testIMP_KNN-YTst)^2)
    mseIMP_MICIMP[ns==nlist]=mean(resIMP_MICIMPVec)
    mseIMP_MIMIMP[ns==nlist]=mean(resIMP_MIMIMPVec)
    
    mseCC_S[ns==nlist]=mean((YTst-CC_S)^2)
    mseSVMfullData[ns==nlist]=mean((YTst-SVMfullData)^2)
  } # end of n
  myDF=data.frame(seed = seed,nlist= nlist,
                  mseCC=mseCC,mseWCC_CM=mseWCC_CM,mseWCC_MM=mseWCC_MM,
                  mseDR_CMCIMP=mseDR_CMCIMP,mseDR_CMMIMP=mseDR_CMMIMP,
                  mseDR_MMCIMP=mseDR_MMCIMP,mseDR_MMMIMP=mseDR_MMMIMP,
                  mseIMP_mean=mseIMP_mean,mseIMP_KNN=mseIMP_KNN,
                  mseIMP_MICIMP=mseIMP_MICIMP,mseIMP_MIMIMP=mseIMP_MIMIMP,
                  mseCC_S=mseCC_S,mseSVMfullData=mseSVMfullData)
  return(myDF)
} # end of function 


 
