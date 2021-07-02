# setwd("/data/home/tiantian.liu/MisCov/subFun/setting4")
setwd("C:/Users/liutiantian/Desktop/setting4")
##parameters
library(kernlab)
library(modopt.matlab)
n=100;missing=1;
pInd=17;pR=6;alpha=-1/2;beta=1;gamma=-1/10;delta=20
omega11=-1/8;eta11=-2#-1#-1/3;
eta21=1/8;omega12=1/2#2/3#1;
eta12=0#1;
eta22=0#1;
B=5

gmFun=function(seed,nlist = c(100,200)){
  library(DMwR)
  library(e1071)
  source('misCovLearner.R')
  source('MImodels.R')
  source('pathes.R')
  source('tuningParamSet1.R')
  source('myCVMeasure.R')
  source('generalmissing.R')
  
  nTst=10000
  misIndicTst=0
  set.seed(10000)
  dataTst=generalMissing(nTst,pInd,pR,misIndicTst,gamma,delta,alpha,beta,
                         omega11,omega12,eta11,eta12,eta21,eta22)
  p=20  # dimension of covariates
  px=14

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
    misCovDat=generalMissing(nTrain,pInd,pR,1,gamma,delta,alpha,beta,
                             omega11,omega12,eta11,eta12,eta21,eta22)
    ## Step 1: Make task
    regr.task=makeRegrTask(id="misCov",data=misCovDat,target = "Y")

    ## Step 2: Make learner for corss validation
    # 5 flods for cv
    ctrl = makeTuneControlGrid()
    rdesc = makeResampleDesc("CV", iters = 5L)

    ## CC method
    discrete_psCC=tuningParamSet1(px,"CC","RBF",psPath0,impPath0,B)
    resCC= tuneParams("regr.misCovSVMg", task = regr.task, resampling = rdesc,
                      measures=my.cv.measure, par.set = discrete_psCC, control = ctrl)
    par.valsCC= resCC$x # get the best parameter of lambda and sigma
    par.valsCC[["testPurpose"]]= "testing"
    lrnCC= setHyperPars(makeLearner("regr.misCovSVMg"), par.vals = par.valsCC) # Make the learner
    # Step 3: Train the learner
    trainAlgoCC= train(lrnCC,regr.task)
    # Step 4: Test the learner using the testing data set
    testCC=predict(trainAlgoCC,newdata=dataTst)


    ## WCC method (if propensity score is correctly specified, psPath1; misspecified, psPath0).
    discrete_psWCC_CM=tuningParamSet1(px,"WCC","RBF",psPath1,impPath0,B)
    resWCC_CM= tuneParams("regr.misCovSVMg", task = regr.task, resampling = rdesc,
                          measures=my.cv.measure, par.set = discrete_psWCC_CM,
                          control = ctrl)
    par.valsWCC_CM= resWCC_CM$x # get the best parameter of lambda and sigma
    par.valsWCC_CM[["testPurpose"]]= "testing"
    lrnWCC_CM= setHyperPars(makeLearner("regr.misCovSVMg"), par.vals =par.valsWCC_CM) # Make learner
    trainAlgoWCC_CM= train(lrnWCC_CM,regr.task) # training
    testWCC_CM=predict(trainAlgoWCC_CM,newdata=dataTst) # testing

    ## misspecified the propensity score model
    discrete_psWCC_MM=tuningParamSet1(px,"WCC","RBF",psPath0,impPath0,B)
    resWCC_MM= tuneParams("regr.misCovSVMg", task = regr.task, resampling = rdesc,
                          measures=my.cv.measure, par.set = discrete_psWCC_MM, control = ctrl)
    par.valsWCC_MM= resWCC_MM$x
    par.valsWCC_MM[["testPurpose"]]= "testing"
    lrnWCC_MM= setHyperPars(makeLearner("regr.misCovSVMg"), par.vals =par.valsWCC_MM) # Make learner
    trainAlgoWCC_MM= train(lrnWCC_MM,regr.task) # training
    testWCC_MM=predict(trainAlgoWCC_MM,newdata=dataTst) # testing

    ## both correctly specified
    discrete_psDR_CMCIMP=tuningParamSet1(px,"DR","RBF",psPath1,impPath1,B) # PSmodel=1, Regmodel=1
    resDR_CMCIMP= tuneParams("regr.misCovSVMg", task = regr.task, resampling = rdesc,
                             measures=my.cv.measure, par.set = discrete_psDR_CMCIMP, control = ctrl)
    par.valsDR_CMCIMP= resDR_CMCIMP$x # get the best parameter of lambda and sigma
    par.valsDR_CMCIMP[["testPurpose"]]= "testing"
    lrnDR_CMCIMP= setHyperPars(makeLearner("regr.misCovSVMg"), par.vals = par.valsDR_CMCIMP) # Make learner
    trainAlgoDR_CMCIMP= train(lrnDR_CMCIMP,regr.task) # trainging
    testDR_CMCIMP=predict(trainAlgoDR_CMCIMP,newdata=dataTst) # testing


    ## propensity score correctly specified, imputation misspecified
    discrete_psDR_CMMIMP=tuningParamSet1(px,"DR","RBF",psPath1,impPath0,B) # PSmodel=1, Regmodel=0
    resDR_CMMIMP= tuneParams("regr.misCovSVMg", task = regr.task, resampling = rdesc,
                             measures=my.cv.measure, par.set = discrete_psDR_CMMIMP, control = ctrl)
    par.valsDR_CMMIMP= resDR_CMMIMP$x # get the best parameter of lambda and sigma
    par.valsDR_CMMIMP[["testPurpose"]]= "testing"
    lrnDR_CMMIMP= setHyperPars(makeLearner("regr.misCovSVMg"), par.vals = par.valsDR_CMMIMP) # Make learner
    trainAlgoDR_CMMIMP= train(lrnDR_CMMIMP,regr.task) # trainging
    testDR_CMMIMP=predict(trainAlgoDR_CMMIMP,newdata=dataTst) # testing

    ## propensity score misspecified, imputation correctly specified
    discrete_psDR_MMCIMP=tuningParamSet1(px,"DR","RBF",psPath0,impPath1,B)
    resDR_MMCIMP= tuneParams("regr.misCovSVMg", task = regr.task, resampling = rdesc,
                             measures=my.cv.measure, par.set = discrete_psDR_MMCIMP, control = ctrl)
    par.valsDR_MMCIMP= resDR_MMCIMP$x # get the best parameter of lambda and sigma
    par.valsDR_MMCIMP[["testPurpose"]]= "testing"
    lrnDR_MMCIMP= setHyperPars(makeLearner("regr.misCovSVMg"), par.vals = par.valsDR_MMCIMP) # Make learner
    trainAlgoDR_MMCIMP= train(lrnDR_MMCIMP,regr.task) # trainging
    testDR_MMCIMP=predict(trainAlgoDR_MMCIMP,newdata=dataTst) # testing

    ## both misspecified
    discrete_psDR_MMMIMP=tuningParamSet1(px,"DR","RBF",psPath0,impPath0,B)
    resDR_MMMIMP= tuneParams("regr.misCovSVMg", task = regr.task, resampling = rdesc,
                             measures=my.cv.measure, par.set = discrete_psDR_MMMIMP, control = ctrl)
    par.valsDR_MMMIMP= resDR_MMMIMP$x # get the best parameter of lambda and sigma
    par.valsDR_MMMIMP[["testPurpose"]]= "testing"
    lrnDR_MMMIMP= setHyperPars(makeLearner("regr.misCovSVMg"), par.vals = par.valsDR_MMMIMP) # Make learner
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
      , cost = p^(-6:6)
    )
    SVMfullData=predict(SVMfullData, dataTst[,1:p])
    SVMfullData=2*(as.numeric(SVMfullData)-1)-1

    ## mean inputation

    Index1=(misCovDat$R==0)*(misCovDat$R1==0) ## x16,x17,x19,x20
    Index2=(misCovDat$R==0)*(misCovDat$R2==0) ## x15,x18
    Index=(misCovDat$R==0)-Index1-Index2
    misCovDat$x16[Index1==1]=NA;misCovDat$x17[Index1==1]=NA
    misCovDat$x19[Index1==1]=NA;misCovDat$x20[Index1==1]=NA
    misCovDat$x15[Index2==1]=NA;misCovDat$x18[Index2==1]=NA
    misCovDat_meanIMP=misCovDat
    misCovDat_meanIMP$x15[is.na(misCovDat_meanIMP$x15)]=mean(misCovDat_meanIMP$x15[!is.na(misCovDat_meanIMP$x15)])
    misCovDat_meanIMP$x16[is.na(misCovDat_meanIMP$x16)]=mean(misCovDat_meanIMP$x16[!is.na(misCovDat_meanIMP$x16)])
    misCovDat_meanIMP$x17[is.na(misCovDat_meanIMP$x17)]=mean(misCovDat_meanIMP$x17[!is.na(misCovDat_meanIMP$x17)])
    misCovDat_meanIMP$x18[is.na(misCovDat_meanIMP$x18)]=mean(misCovDat_meanIMP$x18[!is.na(misCovDat_meanIMP$x18)])
    misCovDat_meanIMP$x19[is.na(misCovDat_meanIMP$x19)]=mean(misCovDat_meanIMP$x19[!is.na(misCovDat_meanIMP$x19)])
    misCovDat_meanIMP$x20[is.na(misCovDat_meanIMP$x20)]=mean(misCovDat_meanIMP$x20[!is.na(misCovDat_meanIMP$x20)])

    trainAlgoIMP_mean <- best.svm(
      x = misCovDat_meanIMP[,1:p]
      , y = factor(misCovDat_meanIMP$Y)
      , type = "C-classification"
      , kernel = "radial"
      , cost =p^(-6:6)#1/(2*nTrain*10^(-3:1)) # 2^(-6:6)
    )

    testIMP_mean=predict(trainAlgoIMP_mean, dataTst[,1:p])
    testIMP_mean=2*(as.numeric(testIMP_mean)-1)-1

    misCovDat_KnnIMP=knnImputation(misCovDat,k=5)
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
    V=misCovDat[,(px+1):(pTotal-5)] # the popential missing covariates
    Z=cbind(X,V) # the covariates
    R=misCovDat[,(pTotal-4)]
    R1=misCovDat[,(pTotal-3)]
    R2=misCovDat[,(pTotal-2)]   #  missing indicator
    Y=misCovDat[,pTotal] # the response (-1 and 1)
    n=nrow(misCovDat) # sample size of the traing data
    YCC=Y[R==1] # Y corresponding complete cases
    ZCC=Z[R==1,] # covariates corresponding complete cases
    XCC=Z[R==1,1:px] # Y corresponding complete cases
    VCC=Z[R==1,(px+1):p] # covariates corresponding complete cases
    Index1=(R==0)*(R1==0) ## x16,x17,x19,x20
    Index2=(R==0)*(R2==0) ## x15,x18
    Index=(R==0)-Index1-Index2

    dataIC=cbind(Z[Index==1,],Y[Index==1]) # observed incomplete data
    dataPC2=cbind(Z[Index1==1,],Y[Index1==1])
    dataPC3=cbind(Z[Index2==1,],Y[Index2==1])
    dataCC=cbind(XCC,VCC,YCC) # complete data
    B=5 ## imputation times

    ## imputation under model1
    IMP_MICIMP1=modelImp_MIcorrect(XCC,YCC,VCC)
    IMP_MICIMP2=modelImp_MIcorrect(cbind(XCC,VCC[,1],VCC[,4]),YCC,VCC[,c(2:3,5:6)])
    IMP_MICIMP3=modelImp_MIcorrect(cbind(XCC,VCC[,2:3],VCC[,5:6]),YCC,VCC[,c(1,4)])
    resIMP_MICIMPVec=rep(0,B)
    for(i in 1:B){
      imputeData1=t(apply(dataIC,1,IMP_MICIMP1))
      imputeData2=t(apply(dataPC2,1,IMP_MICIMP2))
      imputeData3=t(apply(dataPC3,1,IMP_MICIMP3))
      fullData=rbind(dataCC,imputeData1,imputeData2,imputeData3)
      fullData=as.data.frame(fullData)
      cloName=c(paste0("x",1:20),"Y")
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
    IMP_MIMIMP2=modelImp_MImiss(cbind(XCC,VCC[,1],VCC[,4]),YCC,VCC[,c(2:3,5:6)])
    IMP_MIMIMP3=modelImp_MImiss(cbind(XCC,VCC[,2:3],VCC[,5:6]),YCC,VCC[,c(1,4)])
    resIMP_MIMIMPVec=rep(0,B)
    for(i in 1:B){
      imputeData1=t(apply(dataIC,1,IMP_MIMIMP1))
      imputeData2=t(apply(dataPC2,1,IMP_MIMIMP2))
      imputeData3=t(apply(dataPC3,1,IMP_MIMIMP3))
      fullData=rbind(dataCC,imputeData1,imputeData2,imputeData3)
      fullData=as.data.frame(fullData)
      cloName=c(paste0("x",1:20),"Y")
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
    rate_pos=mean(misCovDat[,24]==1)

    YCC=testCC$data[,2]
    YCC[YCC==0]=2*(rbinom(sum(YCC==0),1,rate_pos))-1
    mseCC[ns==nlist]=mean((YTst-YCC)^2)

    YWCC_CM=testWCC_CM$data[,2]
    YWCC_CM[YWCC_CM==0]=2*(rbinom(sum(YWCC_CM==0),1,rate_pos))-1
    mseWCC_CM[ns==nlist]=mean((YTst-YWCC_CM)^2)

    YWCC_MM=testWCC_MM$data[,2]
    YWCC_MM[YWCC_MM==0]=2*(rbinom(sum(YWCC_MM==0),1,rate_pos))-1
    mseWCC_MM[ns==nlist]=mean((YTst-YWCC_MM)^2)

    YDR_CMCIMP=testDR_CMCIMP$data[,2]
    YDR_CMCIMP[YDR_CMCIMP==0]=2*(rbinom(sum(YDR_CMCIMP==0),1,rate_pos))-1
    mseDR_CMCIMP[ns==nlist]=mean((YTst-YDR_CMCIMP)^2)

    YDR_CMMIMP=testDR_CMMIMP$data[,2]
    YDR_CMMIMP[YDR_CMMIMP==0]=2*(rbinom(sum(YDR_CMMIMP==0),1,rate_pos))-1
    mseDR_CMMIMP[ns==nlist]=mean((YTst-YDR_CMMIMP)^2)
    

    YDR_MMCIMP=testDR_MMCIMP$data[,2]
    YDR_MMCIMP[YDR_MMCIMP==0]=2*(rbinom(sum(YDR_MMCIMP==0),1,rate_pos))-1
    mseDR_MMCIMP[ns==nlist]=mean((YTst-YDR_MMCIMP)^2)
    

    YDR_MMMIMP=testDR_MMMIMP$data[,2]
    YDR_MMMIMP[YDR_MMMIMP==0]=2*(rbinom(sum(YDR_MMMIMP==0),1,rate_pos))-1
    mseDR_MMMIMP[ns==nlist]=mean((YTst-YDR_MMMIMP)^2)


    mseIMP_mean[ns==nlist]=mean((testIMP_mean-YTst)^2)
    mseIMP_KNN[ns==nlist]=mean((testIMP_KNN-YTst)^2)
    mseIMP_MICIMP[ns==nlist]=mean(resIMP_MICIMPVec)
    mseIMP_MIMIMP[ns==nlist]=mean(resIMP_MIMIMPVec)

    mseCC_S[ns==nlist]=mean((YTst-CC_S)^2)
    mseSVMfullData[ns==nlist]=mean((YTst-SVMfullData)^2)
  }
  myDF=data.frame(seed = seed,mseCC=mseCC,
                  mseWCC_CM=mseWCC_CM,mseWCC_MM=mseWCC_MM,
                  mseDR_CMCIMP=mseDR_CMCIMP,mseDR_MMCIMP=mseDR_MMCIMP,
                  mseDR_CMMIMP=mseDR_CMMIMP,mseDR_MMMIMP=mseDR_MMMIMP,
                  mseIMP_mean=mseIMP_mean,mseIMP_KNN=mseIMP_KNN,
                  mseIMP_MICIMP=mseIMP_MICIMP,mseIMP_MIMIMP=mseIMP_MIMIMP,
                  mseCC_S=mseCC_S,mseSVMfullData=mseSVMfullData)
  return(myDF)
}




