# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

#Tenha a planilha na mesma pasta que o código!
tabelaCotas=pd.read_excel(os.path.join(os.path.dirname(__file__), "Cotas.xlsx"),header=None).to_numpy()
#Linhas em 0 (↓): eixo x
#Valores no encontro: eixo y
#Colunas em 0 (→): eixo z

#Para fins de facilitar a modelagem das splines, excluiremos a primeira linha
#d'água e a última baliza. Isto equivale a uma aproximação para um casco reto
#nestas regiões.
def eliminateZeros(tabelaCotas):
    tabelaCotas=tabelaCotas[:len(tabelaCotas)-1,:]
    tabelaCotas=np.delete(tabelaCotas,1,1)
    return tabelaCotas

def interpolSideways(tabelaCotas):
    
    #storeS é reponsável por ir armazenando os valores de newLine que são
    #valores novos de cada linha além dos já existentes
    storeS=[]
    
    #Formando a primera linha da nova matriz
    newLine=[tabelaCotas[0,0],tabelaCotas[0,1]]
    for j in range(2,len(tabelaCotas[0])):
        halfVal=tabelaCotas[0,j]-(tabelaCotas[0,j]-tabelaCotas[0,j-1])/2
        newLine.append(halfVal)
        newLine.append(tabelaCotas[0,j])
    
    storeS.append(newLine)    
       
    #Formando todas as novas linhas da nova matriz. newLine é resetado toda
    #vez que este loop é executado.
    for i in range(1,len(tabelaCotas)):
        
        #Calculo dos h's
        storeH=[]
        beginCounter=len(tabelaCotas[0])
        for j in range(2,len(tabelaCotas[0])):
            h=tabelaCotas[0,j]-tabelaCotas[0,j-1]
            if tabelaCotas[i,j]-tabelaCotas[i,j-1]!=0:
                storeH.append(h)
                beginCounter-=1
        
        #Calculo da matriz A e da matrz b
        A=np.zeros((len(storeH)-1,len(storeH)+1))
        b=np.zeros((len(storeH)-1))
        for m in range(len(storeH)-1):
            A[m,m]=storeH[m]
            A[m,m+1]=2*(storeH[m]+storeH[m+1])
            A[m,m+2]=storeH[m+1]
            
            b[m]=(tabelaCotas[i,m+beginCounter+1]-tabelaCotas[i,m+beginCounter])/storeH[m+1]-(tabelaCotas[i,m+beginCounter]-tabelaCotas[i,m+beginCounter-1])/storeH[m]
        
        A=A[:,1:len(A[0])-1] #Removendo primeira e última coluna para a operação
        b=6*b
        
        #Calculo da matriz x com os coeficientes g
        matrixX=np.linalg.solve(A,b)
        matrixX=np.append(matrixX, 0.0) #Adicionando o primeiro e último zero que foram retirados
        matrixX=np.insert(matrixX, 0, 0.0, axis=0)
        
        #Calculo das splines entre cada ponto e seu valor
        newLine=[tabelaCotas[i,0],tabelaCotas[i,1]]
        zeroCounter=2
        for k in range(2,len(tabelaCotas[0])):
            
            if tabelaCotas[i,k]-tabelaCotas[i,k-1]==0:
                newLine.append(0.0)
                newLine.append(tabelaCotas[i,k])
                zeroCounter+=1
            else:
                #Calculo dos coeficientes
                a=(matrixX[k-zeroCounter+1]-matrixX[k-zeroCounter])/(6*storeH[k-zeroCounter])
                b=matrixX[k-zeroCounter+1]/2
                c=(tabelaCotas[i,k]-tabelaCotas[i,k-1])/storeH[k-zeroCounter]+(2*storeH[k-zeroCounter]*matrixX[k-zeroCounter+1]+matrixX[k-zeroCounter]*storeH[k-zeroCounter])/6
                d=tabelaCotas[i,k]
                
                #Calculo do ponto entre dois conhecidos e append para ir construindo a nova linha
                halfVal=tabelaCotas[0,k]-(tabelaCotas[0,k]-tabelaCotas[0,k-1])/2
                s=a*(halfVal-tabelaCotas[0,k])**3+b*(halfVal-tabelaCotas[0,k])**2+c*(halfVal-tabelaCotas[0,k])+d

                newLine.append(s)
                newLine.append(tabelaCotas[i,k])
        
        storeS.append(newLine)
        
    #Transformando a lista final em matriz e atribuindo a tabela de cotas
    tabelaCotas=np.asarray(storeS)
    return tabelaCotas

def interpolDownwards(tabelaCotas):
    storeS=[]
    
    #Formando a primera coluna da nova matriz
    newColumn=[tabelaCotas[0,0],tabelaCotas[1,0]]
    for i in range(2,len(tabelaCotas)):
        halfVal=tabelaCotas[i,0]-(tabelaCotas[i,0]-tabelaCotas[i-1,0])/2
        newColumn.append(halfVal)
        newColumn.append(tabelaCotas[i,0])
    
    storeS.append(newColumn)    
       
    #Formando todas as novas colunas da nova matriz. newColumn é resetado toda
    #vez que este loop é executado.
    for j in range(1,len(tabelaCotas[0])):
        
        #Calculo dos h's
        storeH=[]
        beginCounter=len(tabelaCotas)
        for i in range(2,len(tabelaCotas)):
            h=tabelaCotas[i,0]-tabelaCotas[i-1,0]
            
            #Adicionamos apenas os h's diferentes de zero para nossa matriz
            #storeH para que calculemos apenas com no máximo 1 ponto igual a zero
            if tabelaCotas[i,j]-tabelaCotas[i-1,j]!=0:
                storeH.append(h)           
                beginCounter-=1
            
            #Para o caso de cada coluna precisamos fazer esta condição para não
            #obtermos erro no beginCounter
            elif tabelaCotas[i,j]-tabelaCotas[i-1,j]==0 and i>len(tabelaCotas)/2:
                beginCounter-=1
            
        #Calculo da matriz A e da matrz b
        A=np.zeros((len(storeH)-1,len(storeH)+1))
        b=np.zeros((len(storeH)-1))
        for m in range(len(storeH)-1):
            A[m,m]=storeH[m]
            A[m,m+1]=2*(storeH[m]+storeH[m+1])
            A[m,m+2]=storeH[m+1]
            
            b[m]=(tabelaCotas[m+beginCounter+1,j]-tabelaCotas[m+beginCounter,j])/storeH[m+1]-(tabelaCotas[m+beginCounter,j]-tabelaCotas[m+beginCounter-1,j])/storeH[m]
        A=A[:,1:len(A[0])-1] #Removendo primeira e última coluna para a operação
        b=6*b
        
        #Calculo da matriz x com os coeficientes g
        matrixX=np.linalg.solve(A,b)
        matrixX=np.append(matrixX, 0.0) #Adicionando o primeiro e último zero que foram retirados
        matrixX=np.insert(matrixX, 0, 0.0, axis=0)
     
        #Calculo das splines entre cada ponto e seu valor
        newColumn=[tabelaCotas[0,j],tabelaCotas[1,j]]
        zeroCounter=2
        for k in range(2,len(tabelaCotas)):
            
            if tabelaCotas[k,j]-tabelaCotas[k-1,j]==0:
                newColumn.append(0.0)
                newColumn.append(tabelaCotas[k,j])
                zeroCounter+=1
            else:
                #Calculo dos coeficientes
                a=(matrixX[k-zeroCounter+1]-matrixX[k-zeroCounter])/(6*storeH[k-zeroCounter])
                b=matrixX[k-zeroCounter+1]/2
                c=(tabelaCotas[k,j]-tabelaCotas[k-1,j])/storeH[k-zeroCounter]+(2*storeH[k-zeroCounter]*matrixX[k-zeroCounter+1]+matrixX[k-zeroCounter]*storeH[k-zeroCounter])/6
                d=tabelaCotas[k,j]
            
                #Calculo do ponto entre dois pontos conhecidos e append para ir 
                #construindo a nova coluna
                halfVal=tabelaCotas[k,0]-(tabelaCotas[k,0]-tabelaCotas[k-1,0])/2
                s=a*(halfVal-tabelaCotas[k,0])**3+b*(halfVal-tabelaCotas[k,0])**2+c*(halfVal-tabelaCotas[k,0])+d
                newColumn.append(s)
                newColumn.append(tabelaCotas[k,j])
        
        storeS.append(newColumn)
    
    #Transformando a lista em matriz e a tranpondo para obter ela em forma de colunas
    tabelaCotas=np.asarray(storeS).transpose()
    return tabelaCotas

def nInterpolBoth(tabelaCotas, n):
    for i in range(1,n+1):
        tabelaCotas=interpolDownwards(interpolSideways(tabelaCotas))

    return tabelaCotas

def nInterpolDownwards(tabelaCotas, n):
    for i in range(1,n+1):
        tabelaCotas=interpolDownwards(tabelaCotas)

    return tabelaCotas

def nInterpolSideways(tabelaCotas, n):
    for i in range(1,n+1):
        tabelaCotas=interpolSideways(tabelaCotas)

    return tabelaCotas
    
#-------------Definição do Calado e seu índice na tabela de Cotas-----------------------#

def chooseCalado(tabelaCotas, calado):
    
    caladoIndex=0
    for j in range(len(tabelaCotas[0])):
        if tabelaCotas[0,j]==calado:
            caladoIndex=j
        elif tabelaCotas[0,j]<calado and tabelaCotas[0,j+1]>calado:
            caladoIndex=j
    
    caladoTab=tabelaCotas[0,caladoIndex]
    print("\nCalado Considerado:",caladoTab)
    return caladoIndex, caladoTab

#------------Cálculo das propriedades hidroestaticas sem zeros--------------------------#
def hidroProps(tabelaCotas, caladoIndex, caladoTab):
    #i representa as linhas
    #j representa as colunas
    storeVectorA=[]
    storeScalarA=[]
    storeC=[]
    #--------------------Cálculo dos Painéis Laterais---------------------------- #   
    for i in range(1,len(tabelaCotas)-1):
        for j in range(2,caladoIndex+1):
            
            #Vetores painéis
            v1=np.array([0,tabelaCotas[i,j]-tabelaCotas[i,j-1],tabelaCotas[0,j]-tabelaCotas[0,j-1]]) #p2-p1
            v2=np.array([tabelaCotas[i+1,0]-tabelaCotas[i,0],tabelaCotas[i+1,j-1]-tabelaCotas[i+1,j],0]) #p4-p1
            v3=np.array([0,tabelaCotas[i+1,j-1]-tabelaCotas[i+1,j],tabelaCotas[0,j-1]-tabelaCotas[0,j]]) #p4-p3
            v4=np.array([tabelaCotas[i,0]-tabelaCotas[i+1,0],tabelaCotas[i,j]-tabelaCotas[i+1,j],0]) #p2-p3
            
            #Calculo vetor A e C do painel sendo analisado
            currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            currentC=np.array([(2*tabelaCotas[i,0]+2*tabelaCotas[i+1,0])/4,(tabelaCotas[i,j]+tabelaCotas[i,j-1]+tabelaCotas[i+1,j-1]+tabelaCotas[i+1,j])/4,(2*tabelaCotas[0,j]+2*tabelaCotas[0,j-1])/4])
            
            #Guardando A em forma vetor e escalar e C vetor em listas
            storeVectorA.append(currentA.copy())
            storeScalarA.append(np.linalg.norm(currentA))
            storeC.append(currentC)
            
            #Para o outro lado
            v1=np.array([tabelaCotas[i+1,0]-tabelaCotas[i,0],(-tabelaCotas[i+1,j-1])-(-tabelaCotas[i,j-1]),0]) #p2-p1
            v2=np.array([0,(-tabelaCotas[i,j])-(-tabelaCotas[i,j-1]),tabelaCotas[0,j]-tabelaCotas[0,j-1]]) #p4-p1
            v3=np.array([tabelaCotas[i,0]-tabelaCotas[i+1,0],(-tabelaCotas[i,j])-(-tabelaCotas[i+1,j]),0]) #p4-p3
            v4=np.array([0,(-tabelaCotas[i+1,j-1])-(-tabelaCotas[i+1,j]),tabelaCotas[0,j-1]-tabelaCotas[0,j]]) #p2-p3
            
            currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            currentC=np.array([(2*tabelaCotas[i,0]+2*tabelaCotas[i+1,0])/4,(-tabelaCotas[i,j]-tabelaCotas[i,j-1]-tabelaCotas[i+1,j-1]-tabelaCotas[i+1,j])/4,(2*tabelaCotas[0,j]+2*tabelaCotas[0,j-1])/4])
            
            storeVectorA.append(currentA.copy())
            storeScalarA.append(np.linalg.norm(currentA))
            storeC.append(currentC.copy())
            
    #----------------------Cálculo dos painéis da Popa----------------------------#
    storeVectorPopA=[]
    storeScalarPopA=[]
    storePopC=[]
    
    for j in range(1,caladoIndex):
        v1=np.array([0,0,tabelaCotas[0,j+1]-tabelaCotas[0,j]])#p2-p1
        v2=np.array([0,tabelaCotas[1,j],0])#p4-p1
        v3=np.array([0,tabelaCotas[1,j]-tabelaCotas[1,j+1],tabelaCotas[0,j]-tabelaCotas[0,j+1]])#p4-p3
        v4=np.array([0,0-tabelaCotas[1,j+1],0])#p2-p3
        
        currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        currentC=np.array([0,(tabelaCotas[1,j]+tabelaCotas[1,j+1])/4,(2*tabelaCotas[0,j]+2*tabelaCotas[0,j+1])/4])
        
        storeVectorPopA.append(currentA.copy())
        storeScalarPopA.append(np.linalg.norm(currentA))
        storePopC.append(currentC.copy())
        
        #Do outro lado
        v1=np.array([0,-tabelaCotas[1,j],0])#p2-p1
        v2=np.array([0,0,tabelaCotas[0,j+1]-tabelaCotas[0,j]])#p4-p1
        v3=np.array([0,tabelaCotas[1,j+1],0])#p4-p3
        v4=np.array([0,-tabelaCotas[1,j]-(-tabelaCotas[1,j+1]),tabelaCotas[0,j]-tabelaCotas[0,j+1]])#p2-p3
        
        currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        currentC=np.array([0,(-tabelaCotas[1,j]-tabelaCotas[1,j+1])/4,(2*tabelaCotas[0,j]+2*tabelaCotas[0,j+1])/4])
        
        storeVectorPopA.append(currentA.copy())
        storeScalarPopA.append(np.linalg.norm(currentA))
        storePopC.append(currentC.copy())
        
    #-----------------------Cálculo dos painéis do Topo-------------------------------#
    
    storeVectorTopA=[]
    storeScalarTopA=[]
    storeTopC=[]
    storeBal=[]
    
    #Formando a matriz do plano da linha d'água matrixYWL
    matrixYWL=tabelaCotas[1:,:caladoIndex+1] #É extraído a primeira coluna com os valores de x até a coluna com a spline da linha d'água do calado
    matrixYWL=np.delete(matrixYWL,np.s_[1:len(matrixYWL[0])-1],axis=1) #Deletamos todas as colunas entre essas duas colunas
    
    #Loop para obter valores igualmente espaçados do y=0 até y=spline para cada baliza
    for i in range(len(matrixYWL)):
        currentBal=np.linspace(0,matrixYWL[i,-1],len(tabelaCotas[0])*5) #Aqui é definido a malha do plano de flutuação
        currentBal=np.insert(currentBal,0,matrixYWL[i,0],axis=0)
        storeBal.append(currentBal)
    
    #Transformando lista em matriz    
    matrixYWL=np.asarray(storeBal)
    
    #Cálculo do vetores e painéis
    for i in range(len(matrixYWL)-1):
        for j in range(2,len(matrixYWL[0])):
            v1=np.array([matrixYWL[i+1,0]-matrixYWL[i,0],matrixYWL[i+1,j-1]-matrixYWL[i,j-1],caladoTab])#p2-p1
            v2=np.array([0,matrixYWL[i,j]-matrixYWL[i,j-1],caladoTab])#p4-p1
            v3=np.array([matrixYWL[i,0]-matrixYWL[i+1,0],matrixYWL[i,j]-matrixYWL[i+1,j],caladoTab])#p4-p3
            v4=np.array([0,matrixYWL[i+1,j-1]-matrixYWL[i+1,j],caladoTab])#p2-p3
            
            currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            currentC=np.array([(2*matrixYWL[i,0]+2*matrixYWL[i+1,0])/4,(matrixYWL[i,j]+matrixYWL[i,j-1]+matrixYWL[i+1,j-1]+matrixYWL[i+1,j])/4,caladoTab])
            
            storeVectorTopA.append(currentA.copy())
            storeScalarTopA.append(np.linalg.norm(currentA))
            storeTopC.append(currentC.copy())
    
            #Para o outro lado
            v1=np.array([matrixYWL[i+1,0]-matrixYWL[i,0],-matrixYWL[i+1,j]-(-matrixYWL[i,j]),caladoTab])#p2-p1
            v2=np.array([0,-matrixYWL[i,j-1]-(-matrixYWL[i,j]),caladoTab])#p4-p1
            v3=np.array([matrixYWL[i,0]-matrixYWL[i+1,0],-matrixYWL[i,j-1]-(-matrixYWL[i+1,j-1]),caladoTab])#p4-p3
            v4=np.array([0,-matrixYWL[i+1,j]-(-matrixYWL[i+1,j-1]),caladoTab])#p2-p3
            
            currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            currentC=np.array([(2*matrixYWL[i,0]+2*matrixYWL[i+1,0])/4,(-matrixYWL[i,j]-matrixYWL[i,j-1]-matrixYWL[i+1,j-1]-matrixYWL[i+1,j])/4,caladoTab])
                     
            storeVectorTopA.append(currentA.copy())
            storeScalarTopA.append(np.linalg.norm(currentA))
            storeTopC.append(currentC.copy())
    
    #------------------------Cálculo painéis do Fundo-----------------------------#
    
    storeVectorBotA=[]
    storeScalarBotA=[]
    storeBotC=[]
    
    for i in range(2,len(tabelaCotas)):
        v1=np.array([tabelaCotas[i,0]-tabelaCotas[i-1,0],tabelaCotas[i,1]-tabelaCotas[i-1,1],0])
        v2=np.array([0,-tabelaCotas[i-1,1],0])
        v3=np.array([tabelaCotas[i-1,0]-tabelaCotas[i,0],0,0])#p4-p3
        v4=np.array([0,tabelaCotas[i,1],0])
        
        currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        currentC=np.array([(2*tabelaCotas[i,0]+2*tabelaCotas[i-1,0])/4,(tabelaCotas[i,1]+tabelaCotas[i-1,1])/4,(2*tabelaCotas[0,1])/4])
        
        storeVectorBotA.append(currentA.copy())
        storeScalarBotA.append(np.linalg.norm(currentA))
        storeBotC.append(currentC.copy())
        
        v1=np.array([tabelaCotas[i,0]-tabelaCotas[i-1,0],0,0])#p2-p1
        v2=np.array([0,tabelaCotas[i-1,1],0])#p4-p1
        v3=np.array([tabelaCotas[i-1,0]-tabelaCotas[i,0],-tabelaCotas[i-1,1]-(-tabelaCotas[i,1]),0])#p4-p3
        v4=np.array([0,0-(-tabelaCotas[i,1]),0])#p2-p3
        
        currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        currentC=np.array([(2*tabelaCotas[i,0]+2*tabelaCotas[i-1,0])/4,(-tabelaCotas[i,1]-tabelaCotas[i-1,1])/4,(2*tabelaCotas[0,1])/4])
        
        storeVectorBotA.append(currentA.copy())
        storeScalarBotA.append(np.linalg.norm(currentA))
        storeBotC.append(currentC.copy())
        
    #---------------------Área da superfície Molhada (Sw)---------------------------- #     
    wetS=0
    
    #Todos os painéis multiplicamos por dois para levarmos em conta seu simétrico também
    wetS+=np.sum(storeScalarA)
    wetS+=np.sum(storeScalarBotA)
    wetS+=np.sum(storeScalarPopA)
    
    print("\nSw:",wetS)
    #-------------------Área do plano da linha d'água (Awl)---------------------------#
    
    wetA=0
    storeVectorTopAarray=np.asarray(storeVectorTopA)
    
    wetA=storeVectorTopAarray[:,2].sum(axis=0)
    
    print("\nAw:",wetA) 
    
    
    #------------------------Cálculo de ∇---------------------------------------------#
    xTerm, yTerm, zTerm = 0, 0, 0
    
    for i in range(len(storeVectorA)):
        xTerm+=storeVectorA[i][0]*storeC[i][0]
        yTerm+=storeVectorA[i][1]*storeC[i][1]
        zTerm+=storeVectorA[i][2]*storeC[i][2]    
    
    for i in range(len(storeVectorPopA)):
        xTerm+=storeVectorPopA[i][0]*storePopC[i][0]
        yTerm+=storeVectorPopA[i][1]*storePopC[i][1]
        zTerm+=storeVectorPopA[i][2]*storePopC[i][2]
       
    for i in range(len(storeVectorBotA)):
        xTerm+=storeVectorBotA[i][0]*storeBotC[i][0]
        yTerm+=storeVectorBotA[i][1]*storeBotC[i][1]
        zTerm+=storeVectorBotA[i][2]*storeBotC[i][2]
    
    for i in range(len(storeVectorTopA)):
        xTerm+=storeVectorTopA[i][0]*storeTopC[i][0]
        yTerm+=storeVectorTopA[i][1]*storeTopC[i][1]
        zTerm+=storeVectorTopA[i][2]*storeTopC[i][2]
         
    nabla=(xTerm+yTerm+zTerm)/3
    print("\nNabla(∇):",nabla)
    desloc=nabla*1.025
    print("\nDeslocamento(Δ):",desloc)
    
    #------------------------Cálculo do plano de Flutuação---------------------------#
    
    LCFnumerador, LCFandTCFdenominador,TCFnumerador = 0, 0, 0
    
    for i in range(len(storeVectorTopA)):
        
        LCFnumerador+=(-storeVectorTopA[i][2])*storeTopC[i][0]
        TCFnumerador+=(-storeVectorTopA[i][2])*storeTopC[i][1]   
    
        LCFandTCFdenominador+=(-storeVectorTopA[i][2])
    
    LCF=LCFnumerador/LCFandTCFdenominador
    TCF=TCFnumerador/LCFandTCFdenominador
    
    print("\nLCF, TCF:",LCF, TCF)
    
    #-------------------------Cálculo dos momentos de Inércia-----------------------#
    
    inertiaL, inertiaT = 0, 0
    
    for i in range(len(storeVectorTopA)):
        #Lembrar delft usa a notação trocada
        inertiaT+=(storeVectorTopA[i][2]*(storeTopC[i][1]-TCF)**2)
        inertiaL+=(storeVectorTopA[i][2]*(storeTopC[i][0]-LCF)**2)
    
    print("\nIt, Il:",inertiaT,inertiaL)
    
    #--------------------------Cálculos do Centro de Carena-------------------------#
    
    LCB, TCB, KB = 0, 0, 0
    
    for i in range(len(storeVectorA)):
        LCB+=(storeVectorA[i][0]*storeC[i][0]*storeC[i][0]/2)
        TCB+=(storeVectorA[i][1]*storeC[i][1]*storeC[i][1]/2)
        KB+=(storeVectorA[i][2]*storeC[i][2]*storeC[i][2]/2)
    
    for i in range(len(storeVectorPopA)):
        LCB+=(storeVectorPopA[i][0]*storePopC[i][0]*storePopC[i][0]/2)
        TCB+=(storeVectorPopA[i][1]*storePopC[i][1]*storePopC[i][1]/2)
        KB+=(storeVectorPopA[i][2]*storePopC[i][2]*storePopC[i][2]/2)
    
    for i in range(len(storeVectorBotA)):
        LCB+=(storeVectorBotA[i][0]*storeBotC[i][0]*storeBotC[i][0]/2)
        TCB+=(storeVectorBotA[i][1]*storeBotC[i][1]*storeBotC[i][1]/2)
        KB+=(storeVectorBotA[i][2]*storeBotC[i][2]*storeBotC[i][2]/2)
        
    for i in range(len(storeVectorTopA)):
        LCB+=(storeVectorTopA[i][0]*storeTopC[i][0]*storeTopC[i][0]/2)
        TCB+=(storeVectorTopA[i][1]*storeTopC[i][1]*storeTopC[i][1]/2)
        KB+=(storeVectorTopA[i][2]*storeTopC[i][2]*storeTopC[i][2]/2)
       
    LCB=LCB/nabla
    TCB=TCB/nabla
    KB=KB/nabla
    
    print("\nLCB, TCB, KB:",LCB,TCB,KB)
    
    BML=inertiaL/nabla
    BMT=inertiaT/nabla
    
    print("\nBML, BMT:", BML,BMT)
    
    return (wetS , wetA, nabla, LCF, TCF, inertiaT, inertiaL, LCB, TCB, KB, BML, BMT, desloc)

#------------------------------------------------------------------------------------#    

#Aproximando a nossa tabela de Cotas
tabelaCotas=eliminateZeros(tabelaCotas)
#Fazendo quatro interpolções para cada direação. Após este número de interpolações não há mudanças significativas nos valores hidroestáticos
tabelaCotas=nInterpolBoth(tabelaCotas, 4)

#Loop para obter propriedades por calado
resultsPerCalado=[]
inputs=np.arange(0.75,3.25,0.25)
for i in range(len(inputs)):
    resultsCalado=chooseCalado(tabelaCotas, inputs[i])
    caladoIndex, caladoTab = resultsCalado[0], resultsCalado[1]
    resultsPerCalado.append(hidroProps(tabelaCotas, caladoIndex, caladoTab))

data=[]
inputs=inputs.tolist()
parameters=["Área da superfície molhada (Sw) [m2]","Área do plano de linha d'água (Awl) [m2]","Volume Deslocado [m3]","LCF [m]","TCF [m]","Inércia transversal (IT) [m4]","Inércia longitudinal (IL) [m4]","LCB [m]","TCB [m]","KB [m]","BML [m]","BMT [m]","Deslocamento (Δ) [t]"]

#Loop para obter cada propriedades separadas em listas
for j in range(len(resultsPerCalado[0])):
    currentData=[]
    for i in range(len(resultsPerCalado)):
        currentData.append(resultsPerCalado[i][j])
    
    data.append(currentData)
    
#Loop para plotar todos os gráficos
for i in range(len(parameters)):
    plt.plot(inputs,data[i])
    plt.xlabel("Calado")
    plt.ylabel(str(parameters[i]))
    plt.title(str(parameters[i]) + "  X Calado [m]")
    plt.show()
  
#Loop para printar as listas com o valores de cada propriedade por calado      
print("\nCalados:", inputs)
for i in range(len(data)):
    print("\n"+str(parameters[i])+":",data[i])
    