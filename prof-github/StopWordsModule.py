#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 20:34:24 2022

@author: jfs
"""

import matplotlib.pyplot as plt
import statistics as stat


DeltaX = 5
MaxVal= 99999999999
MaxXaxisValStopWords=4000
FakeSilSizeForZeroSil=15




def Stop_words(NgFreq):
    StopWords={}
    for ngram in NgFreq:
        if(ngram[0] not in StopWords):
            StopWords[ngram[0]]=1
        else:
            StopWords[ngram[0]]+=1
        if(ngram[2] not in StopWords):
            StopWords[ngram[2]]=1
        else:
            StopWords[ngram[2]]+=1
    Silabas_dic(StopWords)
    Ns=StopWords.copy()         
    FullOederedList=dict(sorted(StopWords.items(), key=lambda item: item[1], reverse=True))
    return getStopWords(FullOederedList)



def Num_silabas(Word):
    vogaiscomesemcentos= ["a","e","i","o","u","A","E","I","O","U","à","á","ã","â","é","è","í","ó","õ","ò","Á","À","Ã","Â","Õ","Ô","ê","Ê","y","Y"]
    vogais= ["a","e","i","o","u","A","E","I","O","U","y","Y"]
    nvogais=0; nvogaisAntesAcento=0
    for i in range(len(Word)):
        if (Word[i] in vogaiscomesemcentos):
           nvogais+=1
           if(i < len(Word)-1 and Word[i+1] in vogais):
               nvogaisAntesAcento+= 1
    return nvogais - nvogaisAntesAcento




def Silabas_dic(StopWords):
    for word in StopWords:
        v=StopWords[word]
        sw=Num_silabas(word)
        if(sw == 0):
            sw=FakeSilSizeForZeroSil
        StopWords[word]= v/sw

def getStopWords(FullOederedList):
    ant=MaxVal;StopWords={};
    c=0; 
    cV=[]; cStopWV=[]; StopWordValV=[]; WordsV=[]; ElbowFound=False
    for word in FullOederedList:
        if(c < MaxXaxisValStopWords):
            val=FullOederedList[word]
            cV.append(c); WordsV.append(val)
            if(not ElbowFound):
                if (c%DeltaX == 0):
                    if(ant - val < DeltaX):
                        ant=val; ElbowFound = True
                    else:
                        ant=val
            if(not ElbowFound):           
                StopWordValV.append(val)
                cStopWV.append(c)
                StopWords[word]=True;
        else:    
            return StopWords
        c+=1
   