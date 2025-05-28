#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:30:40 2022

v@author: jfs
"""

import re
from collections import Counter
import weakref
from StopWordsModule import Stop_words
import random
import numpy as np
import datetime

NgramlenghForStopWordSrch=4
FreMin = 2
# size of the sample for p around 0.6 z=1.96 and confidence interval = 0.035
NumRandsForPrecsision=752
ForBidCharSetinRE=(',',':',';','.','!','?','[',']','(',')','<','>','#','|','=','{','}')



class n_gram:
    
    def __init__(self,ngram,freq):
        self.ngram=ngram
        self.freq=freq
        self.QuiSqr=0
        self.OmegaMinusQuiSqr=0
        self.OmegaPlusQuiSqr=0
        self.RelevantQuiSqr=False
        self.Dice=0
        self.OmegaMinusDice=0
        self.OmegaPlusDice=0
        self.RelevantDice=False
        self.MI=0
        self.OmegaMinusMI=0
        self.OmegaPlusMI=0
        self.RelevantMI=False
        self.SCP=0
        self.OmegaMinusSCP=0
        self.OmegaPlusSCP=0
        self.RelevantSCP=False
    
    def calc_scp(self,Dic):
        ng=self.ngram; den=0
        for k in range(len(ng)-1):
            x=ngram2str(0,ng[0:k+1],0); y=ngram2str(0,ng[k+1:len(ng)],0)
            den+= Dic[x].freq * Dic[y].freq
        self.SCP= self.freq**2 / ((1/(len(ng) - 1))*den)
            
  
    def calc_Dice(self,Dic):
        ng=self.ngram; den=0
        for k in range(len(ng)-1):
            x=ngram2str(0,ng[0:k+1],0); y=ngram2str(0,ng[k+1:len(ng)],0)
            den+= Dic[x].freq + Dic[y].freq
        self.Dice= self.freq *2 / ((1/(len(ng) - 1))*den)
        
    def calc_MI(self,Dic,CorpSiz):
        ng=self.ngram; den=0
        for k in range(len(ng)-1):
            x=ngram2str(0,ng[0:k+1],0); y=ngram2str(0,ng[k+1:len(ng)],0)
            den+=  (Dic[x].freq / CorpSiz)  * (Dic[y].freq / CorpSiz)
        self.MI= np.log((self.freq /CorpSiz)  / ((1/(len(ng) - 1))*den))
        
        
    def calc_QuiSqr(self,Dic,CorpSiz):
        ng=self.ngram; Avx=0; Avy=0; Avp=0;
        for k in range(len(ng)-1):
            x=ngram2str(0,ng[0:k+1],0); y=ngram2str(0,ng[k+1:len(ng)],0)
            Avx+=Dic[x].freq; Avy+=Dic[y].freq; Avp+=Dic[x].freq * Dic[y].freq
        Nw=len(ng)-1
        Avx=Avx/Nw;Avy=Avy/Nw; Avp=Avp/Nw; NAvx=CorpSiz - Avx; NAvy=CorpSiz - Avy;
        Num=(self.freq * CorpSiz - Avp)**2;Den = Avx * Avy * NAvx * NAvy
        self.QuiSqr= Num/Den
               
        
    def actualiza_OmegaPlusSCP(self,Dic):
        Pai_SCP=self.SCP
        NgEsqM1St=ngram2str(0,self.ngram,1); NgDirM1St=ngram2str(1,self.ngram,0)
        NgEsqM1=Dic[NgEsqM1St]; NgDirM1=Dic[NgDirM1St]
        CurEsq=NgEsqM1.OmegaPlusSCP; NgEsqM1.OmegaPlusSCP=max(CurEsq,Pai_SCP)
        CurDir=NgDirM1.OmegaPlusSCP; NgDirM1.OmegaPlusSCP=max(CurDir,Pai_SCP)
    
        
    def actualiza_OmegaPlusQuiSqr(self,Dic):
        Pai_QuiSqr=self.QuiSqr
        NgEsqM1St=ngram2str(0,self.ngram,1); NgDirM1St=ngram2str(1,self.ngram,0)
        NgEsqM1=Dic[NgEsqM1St]; NgDirM1=Dic[NgDirM1St]
        CurEsq=NgEsqM1.OmegaPlusQuiSqr; NgEsqM1.OmegaPlusQuiSqr=max(CurEsq,Pai_QuiSqr)
        CurDir=NgDirM1.OmegaPlusQuiSqr; NgDirM1.OmegaPlusQuiSqr=max(CurDir,Pai_QuiSqr)
        
    
    def actualiza_OmegaPlusDice(self,Dic):
        Pai_Dice=self.Dice
        NgEsqM1St=ngram2str(0,self.ngram,1); NgDirM1St=ngram2str(1,self.ngram,0)
        NgEsqM1=Dic[NgEsqM1St]; NgDirM1=Dic[NgDirM1St]
        CurEsq=NgEsqM1.OmegaPlusDice; NgEsqM1.OmegaPlusDice=max(CurEsq,Pai_Dice)
        CurDir=NgDirM1.OmegaPlusDice; NgDirM1.OmegaPlusDice=max(CurDir,Pai_Dice)
        

    def actualiza_OmegaPlusMI(self,Dic):
        Pai_MI=self.MI
        NgEsqM1St=ngram2str(0,self.ngram,1); NgDirM1St=ngram2str(1,self.ngram,0)
        NgEsqM1=Dic[NgEsqM1St]; NgDirM1=Dic[NgDirM1St]
        CurEsq=NgEsqM1.OmegaPlusMI; NgEsqM1.OmegaPlusMI=max(CurEsq,Pai_MI)
        CurDir=NgDirM1.OmegaPlusMI; NgDirM1.OmegaPlusMI=max(CurDir,Pai_MI)


    def actualiza_OmegaMinusSCP(self,Dic,s=ForBidCharSetinRE):
        if (self.freq>= FreMin and not containsAny(self.ngram,s)):
            NgEsqM1St=ngram2str(0,self.ngram,1); NgDirM1St=ngram2str(1,self.ngram,0)
            NgEsqM1=Dic[NgEsqM1St]; NgDirM1=Dic[NgDirM1St]
            self.OmegaMinusSCP = max(NgEsqM1.SCP, NgDirM1.SCP)
            
    def actualiza_OmegaMinusQuiSqr(self,Dic,s=ForBidCharSetinRE):
        if (self.freq>= FreMin and not containsAny(self.ngram,s)):
            NgEsqM1St=ngram2str(0,self.ngram,1); NgDirM1St=ngram2str(1,self.ngram,0)
            NgEsqM1=Dic[NgEsqM1St]; NgDirM1=Dic[NgDirM1St]
            self.OmegaMinusQuiSqr = max(NgEsqM1.QuiSqr, NgDirM1.QuiSqr)
            
    
    def actualiza_OmegaMinusDice(self,Dic,s=ForBidCharSetinRE):
        if (self.freq>= FreMin and not containsAny(self.ngram,s)):
            NgEsqM1St=ngram2str(0,self.ngram,1); NgDirM1St=ngram2str(1,self.ngram,0)
            NgEsqM1=Dic[NgEsqM1St]; NgDirM1=Dic[NgDirM1St]
            self.OmegaMinusDice = max(NgEsqM1.Dice, NgDirM1.Dice)
    
    
    def actualiza_OmegaMinusMI(self,Dic,s=ForBidCharSetinRE):
        if (self.freq>= FreMin and not containsAny(self.ngram,s)):
            NgEsqM1St=ngram2str(0,self.ngram,1); NgDirM1St=ngram2str(1,self.ngram,0)
            NgEsqM1=Dic[NgEsqM1St]; NgDirM1=Dic[NgDirM1St]
            self.OmegaMinusMI = max(NgEsqM1.MI, NgDirM1.MI)
    
    def actualiza_RelevantSCP(self,P,s=ForBidCharSetinRE):
        if (self.freq>= FreMin and not containsAny(self.ngram,s)):
            if len(self.ngram) > 2:
                self.RelevantSCP= self.SCP >= ((1/2)*(self.OmegaMinusSCP **P + self.OmegaPlusSCP **P ))**(1/P)
            else:
                self.RelevantSCP= self.SCP >= self.OmegaPlusSCP
                
    
    def actualiza_RelevantQuiSqr(self,P,s=ForBidCharSetinRE):
        if (self.freq>= FreMin and not containsAny(self.ngram,s)):
            if len(self.ngram) > 2:
                self.RelevantQuiSqr= self.QuiSqr >= ((1/2)*(self.OmegaMinusQuiSqr **P + self.OmegaPlusQuiSqr **P ))**(1/P)
            else:
                self.RelevantQuiSqr= self.QuiSqr >= self.OmegaPlusQuiSqr
                
    
    def actualiza_RelevantDice(self,P,s=ForBidCharSetinRE):
        if (self.freq>= FreMin and not containsAny(self.ngram,s)):
            if len(self.ngram) > 2:
                self.RelevantDice= self.Dice >= ((1/2)*(self.OmegaMinusDice **P + self.OmegaPlusDice **P ))**(1/P)
            else:
                self.RelevantDice= self.Dice >= self.OmegaPlusDice
         
    
    def actualiza_RelevantMI(self,P,s=ForBidCharSetinRE):
        if (self.freq>= FreMin and not containsAny(self.ngram,s)):
            if len(self.ngram) > 2:
                self.RelevantMI= self.MI >= ((1/2)*(self.OmegaMinusMI **P + self.OmegaPlusMI **P ))**(1/P)
            else:
                self.RelevantMI= self.MI >= self.OmegaPlusMI            
                

        
def cria_ngramAndDic(ngram,freq,Dic):
    Dic[ngram2str(0,ngram,0)]=n_gram(ngram,freq)

def ngram2str(OfsetEsq,L,OfsetDir):
    for i in range(OfsetEsq,len(L)-OfsetDir):
        if(i==OfsetEsq):
            s=L[OfsetEsq]
        else:
            s=s+' '+L[i]
    return s


def LocalMaxs(FileName,MaxNgramLength=7,AllMetrics=False,ResSampleForEval=False,P=2):
#    IniTime= datetime.datetime.utcnow()
    PreDic={}; Dic={}
    f = open(FileName, 'r')
    for line in f.readlines():
        elm=replace(line).split(' ')
        for PreNgramSize in range(MaxNgramLength):
            st=pre_n_gram(elm,PreNgramSize+1)
            grava(PreDic,st)
    f.close()
    TriNg={ng:PreDic[ng] for ng in PreDic if len(ng) == NgramlenghForStopWordSrch}
    StopWords=Stop_words(TriNg)  
    CorpSiz=sum(TriNg.values())+ NgramlenghForStopWordSrch - 1
    for ngram in PreDic:
        cria_ngramAndDic(ngram,PreDic[ngram],Dic)
    PreDic={}
    CalculateGlueAndUpdOmegaPlus(Dic,CorpSiz,AllMetrics)
    UpdOmegaMinusAndSetRelevant(Dic,P,MaxNgramLength,StopWords,AllMetrics)
    #
    REsQuiSqr=[ngram for ngram in Dic if Dic[ngram].RelevantQuiSqr]
    REsFile='REsQuiSqr_'+FileName+'_'+str(P)
    WriteOutputFile(REsFile,REsQuiSqr); LenREs=len(REsQuiSqr)
    if(ResSampleForEval):
        SubsetRndForPrecEval=[REsQuiSqr[i**0 * random.randint(0,LenREs-1)] for i in range(NumRandsForPrecsision)]
        PrecisionFile='REsQuiSqr_'+FileName+'_'+str(P)+'_Sample'
        WriteOutputFile(PrecisionFile,SubsetRndForPrecEval)
    #
    if(AllMetrics):
        REsSCP=[ngram for ngram in Dic if Dic[ngram].RelevantSCP]
        REsFile='REsSCP_'+FileName+'_'+str(P)
        WriteOutputFile(REsFile,REsSCP); LenREs=len(REsSCP)
        if(ResSampleForEval):
            SubsetRndForPrecEval=[REsSCP[i**0 * random.randint(0,LenREs-1)] for i in range(NumRandsForPrecsision)]
            PrecisionFile='REsSCP_'+FileName+'_'+str(P)+'_Sample'
            WriteOutputFile(PrecisionFile,SubsetRndForPrecEval)
    #
        REsDice=[ngram for ngram in Dic if Dic[ngram].RelevantDice]
        REsFile='REsDice_'+FileName+'_'+str(P)
        WriteOutputFile(REsFile,REsDice); LenREs=len(REsDice)
        if(ResSampleForEval):
            SubsetRndForPrecEval=[REsDice[i**0 * random.randint(0,LenREs-1)] for i in range(NumRandsForPrecsision)]
            PrecisionFile='REsDice_'+FileName+'_'+str(P)+'_Sample'
            WriteOutputFile(PrecisionFile,SubsetRndForPrecEval)
    #
        REsMI=[ngram for ngram in Dic if Dic[ngram].RelevantMI]
        REsFile='REsMI_'+FileName+'_'+str(P)
        WriteOutputFile(REsFile,REsMI); LenREs=len(REsMI)
        if(ResSampleForEval):
            SubsetRndForPrecEval=[REsMI[i**0 * random.randint(0,LenREs-1)] for i in range(NumRandsForPrecsision)]
            PrecisionFile='REsMI_'+FileName+'_'+str(P)+'_Sample'
            WriteOutputFile(PrecisionFile,SubsetRndForPrecEval)
#    print (datetime.datetime.utcnow() - IniTime) #    return Dic,StopWords

    
def grava(Dic,elm):
    for el in elm:
        if(el not in Dic):
            Dic[el]=1
        else:
            Dic[el]+=1
            
def WriteOutputFile(FileNm,Lst):
    fo=open(FileNm,'w')
    for Ng in Lst:
        st=Ng
        fo.write(st)
        fo.write('\n')
    fo.close()
    
def UpdOmegaMinusAndSetRelevant(Dic,P,MaxNgramLength,StopWords,AllMetrics):
    for ngram in Dic:
        keyref=Dic[ngram]
        if NramGraterThan(ngram,2) and not NramGraterThan(ngram,MaxNgramLength-1) and keyref.ngram[0] not in StopWords and keyref.ngram[len(keyref.ngram)-1] not in StopWords:
            Dic[ngram].actualiza_OmegaMinusQuiSqr(Dic)
            if (AllMetrics):
                Dic[ngram].actualiza_OmegaMinusSCP(Dic)
                Dic[ngram].actualiza_OmegaMinusDice(Dic)
                Dic[ngram].actualiza_OmegaMinusMI(Dic)
        if NramGraterThan(ngram,1) and not NramGraterThan(ngram,MaxNgramLength-1) and keyref.ngram[0] not in StopWords and keyref.ngram[len(keyref.ngram)-1] not in StopWords:
            Dic[ngram].actualiza_RelevantQuiSqr(P)
            if (AllMetrics):
                Dic[ngram].actualiza_RelevantSCP(P)
                Dic[ngram].actualiza_RelevantDice(P)
                Dic[ngram].actualiza_RelevantMI(P)
           
            
def CalculateGlueAndUpdOmegaPlus(Dic,CorpSiz,AllMetrics):
    for ngram in Dic:
        if NramGraterThan(ngram,1):
            NgKey=Dic[ngram]
            NgKey.calc_QuiSqr(Dic,CorpSiz)
            if(AllMetrics):
                NgKey.calc_scp(Dic)
                NgKey.calc_Dice(Dic)
                NgKey.calc_MI(Dic,CorpSiz)
            if NramGraterThan(ngram,2):
                NgKey.actualiza_OmegaPlusQuiSqr(Dic)
                if(AllMetrics):
                    NgKey.actualiza_OmegaPlusSCP(Dic)
                    NgKey.actualiza_OmegaPlusDice(Dic)
                    NgKey.actualiza_OmegaPlusMI(Dic)
            
     
def NramGraterThan(ngram,count):
    return ngram.count(' ') >= count
    
def pre_n_gram(list,n): 
    return [ tuple(list[i:i+n]) for i in range(len(list)-n+1) ]


def containsAny(str, s=ForBidCharSetinRE):
    """ Check whether sequence str contains ANY of the items in set. """
#    return 1 in [c in str for c in s]
    return 1 in [c in w for w in str for c in s]

def replace(line):
    pre= re.sub(';',' ;',re.sub(':',' :', re.sub(',', ' ,',re.sub('}',' }',re.sub('{','{ ',line)))))
    pre=re.sub('\(','( ',re.sub('!',' !',re.sub('>',' >',re.sub('<','< ',re.sub('\.',' .',pre)))))
    pre= re.sub('!',' !',re.sub('\[','[ ',re.sub('\]',' ]',re.sub('\)',' )',pre))))
    return re.sub(r'\s+',' ',re.sub('\n','',re.sub('\?',' ?',pre)))

    
  
