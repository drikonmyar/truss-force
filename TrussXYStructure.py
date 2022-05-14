import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


class Node(object):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            vars(self)[key]=value
class Element(object):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            vars(self)[key]=value
class ElementTrussXY(Element):
    def __init__(self,**kwargs):
        Element.__init__(self,**kwargs)
    
        self.mdof=4
        self.ke=np.zeros((4,4),float)
        self.ok=np.zeros((4,4),float)
        self.ot=np.zeros((4,4),float)
        self.jdofv=np.zeros((4),int)
    
    
    def computeStiffnessMatrix(self,NODE):
        
        self.c=np.zeros((self.mdof,self.mdof),float)
        
        for i in range(self.mdof):
            for j in range(self.mdof):
                self.c[i,j]=0.0
                
        n1=self.nodes[0]
        n2=self.nodes[1]
        self.jdofv[0]=2*n1-1
        self.jdofv[1]=2*n1
        self.jdofv[2]=2*n2-1
        self.jdofv[3]=2*n2
        
        xx=NODE[n2].x-NODE[n1].x
        yy=NODE[n2].y-NODE[n1].y
        
        A=self.A
    
        E=self.E
         
        L=math.sqrt((xx*xx+yy*yy))
        for i in range(4):
            self.ok[i,j]=0.0
            self.ot[i,j]=0.0
        self.ot[i,i]=1.0
        
        C = xx / L
        S = yy / L
        self.ot[0,0]=C
        self.ot[0,1]=S
        self.ot[1,0]=-S
        self.ot[1,1]=C
        self.ot[2,2]=C
        self.ot[2,3]=S
        self.ot[3,2]=-S
        self.ot[3,3]=C
        
        AE_by_L = A * E / L
        
        self.ke[0,0]=      AE_by_L
        self.ke[0,2]=     -AE_by_L
        
        self.ke[2,0]=     -AE_by_L
        self.ke[2,2]=      AE_by_L
        
        for i in range(4):
            for j in range(4):
                self.c[i,j]=0.0
                for k in range(4):
                    self.c[i,j]=self.c[i,j]+self.ot[k,i]*self.ke[k,j]
                    
        for i in range(4):
            for j in range(4):
                self.ok[i,j]=0.0
                for k in range(4):
                    self.ok[i,j]=self.ok[i,j]+self.c[i,k]*self.ot[k,j]
                    
        del self.c
        
    def assembleStiffness(self,gk):
        
        for i in range(self.mdof):
            ii=self.jdofv[i]
            if ii > 0:
                for j in range(self.mdof):
                    jj=self.jdofv[j]
                    if jj > 0:
                        gk[ii-1,jj-1]=gk[ii-1,jj-1]+self.ok[i,j]
                        
    def computeMemberForces(self,NODE):
            self.computeStiffnessMatrix(NODE)
            mfg=[0,0,0,0]
            mfl=[0,0,0,0]
            disp=[0,0,0,0]
            n1=self.nodes[0]
            n2=self.nodes[1]
            N1=NODE[n1]
            N2=NODE[n2]
            disp[0]=N1.Dx
            disp[1]=N1.Dy
            disp[2]=N2.Dx
            disp[3]=N2.Dy
            
            for i in range(4):
                mfg[i]=0.0
                for j in range(4):
                    mfg[i]=mfg[i]+self.ok[i,j]*disp[j]
                    
            for i in range(4):
                mfl[i]=0.0
                for j in range(4):
                    mfl[i]=mfl[i]+self.ot[i,j]*mfg[j]
                    
            self.mfl=mfl
            
class Structure(object):
    
    def __init__(self,**kwargs):
        self.title='Untitled'
        self.numnode=0
        self.numelem=0
        self.NODE=dict()
        self.ELEM=dict()
        self.NODE_LIST=list()
        self.ELEM_LIST=list()
        
    def node(self, **kwargs):
        if 'NODE' not in vars(self):
            self.NODE=dict()
        if 'NODE_LIST' not in vars(self):
            self.NODE_LIST=list()
            
        if 'nid' in kwargs:
            nid=kwargs['nid']
            self.NODE[nid]=Node(**kwargs)
            self.NODE_LIST.append(nid)
            self.numnode=self.numnode+1
            
            
    def element(self, **kwargs):
        if 'ELEM' not in vars(self):
            self.ELEM=dict()
        if 'ELEM_LIST' not in vars(self):
            self.ELEM_LIST=list()
            
        if 'eid' in kwargs:
            eid=kwargs['eid']
            if 'etype' in kwargs:
                self.etype=kwargs['etype']
            if self.etype == 'TrussXY':
                self.ELEM[eid]=ElementTrussXY(**kwargs)
            self.ELEM_LIST.append(eid)
            self.numelem=self.numelem+1
            
    def solve(self,**kwargs):
        self.ndof=2*self.numnode
        self.gk=np.zeros((self.ndof,self.ndof),float)
        self.gp=np.zeros((self.ndof,1),float)
        
        for eid in self.ELEM_LIST:
            self.ELEM[eid].computeStiffnessMatrix(self.NODE)
            self.ELEM[eid].assembleStiffness(self.gk)
            
        for nid in self.NODE_LIST:
            N=self.NODE[nid]
            i1=2*nid-1
            i2=2*nid
            
            if 'Fx' in vars(N):
                self.gp[i1-1,0]=N.Fx
                
            if 'Fy' in vars(N):
                self.gp[i2-1,0]=N.Fy
                
            if 'idx' in vars(N):
                idx=N.idx
                if idx == 1:
                    for i in range(self.ndof):
                        self.gk[i,i1-1]=0.0
                        self.gk[i1-1,i]=0.0
                    self.gk[i1-1,i1-1]=1.0
                    
            if 'idy' in vars(N):
                idy=N.idy
                if idy == 1:
                    for i in range(self.ndof):
                        self.gk[i,i2-1]=0.0
                        self.gk[i2-1,i]=0.0
                    self.gk[i2-1,i2-1]=1.0
                    
        self.disp=np.linalg.solve(self.gk,self.gp)
        
        for nid in self.NODE_LIST:
            i1=2*nid-1
            i2=2*nid
            self.NODE[nid].Dx=self.disp[i1-1,0]
            self.NODE[nid].Dy=self.disp[i2-1,0]
            
        for eid in self.ELEM_LIST:
            self.ELEM[eid].computeMemberForces(self.NODE)
            
    def showStructure(self,**kwargs):
        for key,value in kwargs.items():
            vars(self)[key]=value
        plt.axis((-5.0,55.0,-5.0,55.0))
        ax=plt.gca()
        plt.axis('off')
        for eid in self.ELEM_LIST:
            n1=self.ELEM[eid].nodes[0]
            n2=self.ELEM[eid].nodes[1]
            N1=self.NODE[n1]
            N2=self.NODE[n2]
            p1=[N1.x,N1.y]
            p2=[N2.x,N2.y]
#            print(eid,n1,n2,p1,p2)
            l=mlines.Line2D([N1.x,N2.x],[N1.y,N2.y])
            ax.add_line(l)
        
        plt.show()
        
def problem_303_truss_bridge_xy():
    a=6.0
    h=4.0
    L=a+18
    A_bot=0.1
    A_top=0.1
    A_diag=0.1
    A_vert=0.1
    Iz_bot=0.5e-05
    Iz_top=0.5e-05
    Iz_diag=0.5e-05
    Iz_vert=0.5e-05
    E=2.0e10
    rho=7850
    m_bar=rho*A_bot
    pstr=Structure(etype='TrussXY',title="Truss Bridge - span 24 ft")
    
    pstr.node(nid=1,tagid='L1',x=0.0, y=0, idx=1, idy=1 )
    pstr.node(nid=2,tagid='L2',x=a+2, y=0, Fy=0.0 )
    pstr.node(nid=3,tagid='L3',x=a+10, y=0, Fy=0.0 )
    pstr.node(nid=4,tagid='L4',x=L, y=0, idy=1 )
    
    pstr.node(nid=5,tagid='U2',x=a, y=2, Fy=-8000.0 )
    pstr.node(nid=6,tagid='U3',x=L-6, y=2 )
    pstr.node(nid=7,tagid='U7',x=a+6, y=h, Fy=-4000.0)
    
    pstr.element(eid=1, tagid='L1-U2',etype='TrussXY',nodes=(1,5),
                A=A_diag,E=E,Iz=Iz_diag,rho=rho,m_bar=m_bar)
    pstr.element(eid=2, tagid='L1-L2',nodes=(1,2),
                A=A_bot,E=E,Iz=Iz_bot,rho=rho,m_bar=m_bar)
    pstr.element(eid=3, tagid='L2-U2',nodes=(2,5),
                A=A_diag,E=E,Iz=Iz_diag,rho=rho,m_bar=m_bar)
    pstr.element(eid=4, tagid='U2-U7',nodes=(5,7),
                A=A_diag,E=E,Iz=Iz_diag,rho=rho,m_bar=m_bar)
    pstr.element(eid=5, tagid='L2-U7',nodes=(2,7),
                A=A_diag,E=E,Iz=Iz_diag,rho=rho,m_bar=m_bar)
    pstr.element(eid=6, tagid='L2-L3',nodes=(2,3),
                A=A_bot,E=E,Iz=Iz_bot,rho=rho,m_bar=m_bar)
    pstr.element(eid=7, tagid='U7-U3',nodes=(7,6),
                A=A_diag,E=E,Iz=Iz_diag,rho=rho,m_bar=m_bar)
    pstr.element(eid=8, tagid='L3-U7',nodes=(3,7),
                A=A_diag,E=E,Iz=Iz_diag,rho=rho,m_bar=m_bar)
    pstr.element(eid=9, tagid='L3-U3',nodes=(3,6),
                A=A_diag,E=E,Iz=Iz_diag,rho=rho,m_bar=m_bar)
    pstr.element(eid=10, tagid='L4-U3',nodes=(4,6),
                A=A_diag,E=E,Iz=Iz_diag,rho=rho,m_bar=m_bar)
    pstr.element(eid=11, tagid='L3-L4',nodes=(3,4),
                A=A_bot,E=E,Iz=Iz_bot,rho=rho,m_bar=m_bar)
    
    return pstr

#    pstr.solve()

#    print('Nodes')
#    for nid in pstr.NODE_LIST:
#        N=pstr.NODE[nid]
#        print(nid,N.x,N.y,N.Dx,N.Dy)

#    print('Elements')

#    for eid in pstr.ELEM_LIST:
#        E=pstr.ELEM[eid]
#        print(eid,E.nodes,E.A,E.E,E.mfl[0])
    
