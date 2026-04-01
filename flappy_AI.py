from NNlib import *
from tkinter import *
from tkinter.messagebox import *
import tkinter.font as tkFont
import scipy.special
import numpy
import matplotlib.pyplot
import matplotlib.image
import pickle
import os
import copy
from time import *
from random import *
from math import *

W=1000
H=600
Hplus=150

def colorRGB(r,g,b):
    return '#%02x%02x%02x' % (r,g,b)

def color_grey(n):
    return '#%02x%02x%02x' % (n,n,n)

def delete_obj(c,O):
    for i in range(len(O)):
        c.delete(O[i])

class bird:
    def __init__(self):
        self.x=150
        self.y=randint(100,H-100)
        self.vy=0
        self.r=12
        self.color=color_grey(randint(0,255))
        self.score=0
        #self.brain=neuralNetwork([6,10,1],0) #inputs b.y,b.vy,p.x,p.vx,p.yu,p.yd
        self.brain=neuralNetwork([4,4,1],0) #inputs b.y,b.vy,p.x,p.yu
        self.fitness=0

    def display(self,canvas):
        return [canvas.create_oval(self.x-self.r,self.y+self.r,self.x+self.r,self.y-self.r,fill=self.color)]

    def move(self):
        if(self.y>=H-self.r+self.vy and self.vy<0):
            self.y=H-self.r
            self.vy=0
        elif(self.y<=self.r+self.vy and self.vy>0):
            self.y=self.r
            self.vy=0
        else:
            self.y+=-self.vy
            self.vy-=0.6
            

    def jump(self):
        self.vy=10

def randbird():
    b=bird()
    b.brain.W=[numpy.random.normal(0,pow(b.brain.structure[k+1],-0.5),(b.brain.structure[k+1],b.brain.structure[k])) for k in range(len(b.brain.structure)-1)]
    return b

class pillar:
    def __init__(self,x,yu,yd,vx):
        self.x=x
        self.yu=yu
        self.yd=yd
        self.vx=vx

    def display(self,canvas):
        O=[]
        O+=[canvas.create_rectangle(self.x,self.yu,self.x+100,0,fill="black")]
        O+=[canvas.create_rectangle(self.x,self.yd,self.x+100,H+2,fill="black")]
        return O

    def move(self):
        self.x+=self.vx

def randpillar():
    yu=randint(0,H-251)
    yd=yu+250
    return pillar(W,yu,yd,-4)

def distance(x1,y1,x2,y2):
    return sqrt((x2-x1)**2+(y2-y1)**2)

def detect_collision(b,p):
    if(b.x<p.x and (b.y<p.yu or b.y>p.yd) and b.x>=p.x-b.r):
        return True
    if(b.x<p.x and b.y>p.yu and b.y<p.yd and (distance(b.x,b.y,p.x,p.yu)<=b.r or distance(b.x,b.y,p.x,p.yd)<=b.r)):
        return True
    if(b.x>=p.x and b.x<=p.x+100 and (b.y+b.r>=p.yd or b.y-b.r<=p.yu)):
        return True
    if(b.x>p.x+100 and (distance(b.x,b.y,p.x+100,p.yu)<=b.r or distance(b.x,b.y,p.x+100,p.yd)<=b.r)):
        return True
    return False

def score_display(canvas,b):
    return [canvas.create_line(0,H+2,W,H+2),canvas.create_text(150,(2*H+Hplus)//2,text="Score : "+str(b.score),font=('Times',20))]

def pillar_focus(b,p1,p2):
    if(b.x<=p1.x+100+b.r):
        return p1
    return p2

def calculate_fitness(birds):
    s=0
    for b in birds:
        #b.score-=209
        s+=b.score
    for b in birds:
        b.fitness=b.score/s

def pickOne(birds):
    index=0
    r=numpy.random.uniform(0,1)
    while(r>0):
        r-=birds[index].fitness
        index+=1
    index-=1
    #print("index : ",index)
    parent=birds[index]
    child=bird()
    child.brain=parent.brain
    #print(child.brain.W)
    child.brain.mutate(0.1,0.1)
    #print(child.brain.W)
    return child

root=Tk()
canvas=Canvas(root,width=W,height=H+Hplus,bg="#FFFFFF")
canvas.pack()

origin="C:\\Users\\Raphael\\Desktop\\Programming\\programmation\\python\\NN\\flappy bird\\"

generations=100
birdsize=5
birds=[randbird() for i in range(birdsize)]
birds[0].brain=load("perfect0",origin)
birds[1].brain=load("perfect1",origin)
birds[2].brain=load("perfect2",origin)
birds[3].brain=load("perfect3",origin)
birds[4].brain=load("perfect4",origin)

for g in range(generations):
    B=[birds[i].display(canvas) for i in range(len(birds))]  #display

    pillars=[randpillar(),randpillar(),randpillar()]
    pillars[0].x=W
    pillars[1].x=W+400
    pillars[2].x=W+800
    P=[pillars[i].display(canvas) for i in range(len(pillars))]  #display

    S=score_display(canvas,birds[0])  #display

    canvas.update()   #diplay



    dead_birds=[]
    while(len(birds)>0):
        #####game logic#####
        #sleep(0.01)
       
        for i in range(len(pillars)):
            pillars[i].move()
        if(pillars[0].x<-100):
            pillars=[pillars[1],pillars[2],randpillar()]
            pillars[2].x=1100
            #for i in range(len(pillars)-1,-1,-1):
                #pillars[i].vx=pillars[0].vx-0.1

        new_dead_birds=[]
        for i in range(len(birds)):
            birds[i].move()
            p_focus=pillar_focus(birds[i],pillars[0],pillars[1])
            #think=birds[i].brain.query([birds[i].y/H,birds[i].vy/H,p_focus.x/W,p_focus.vx/20,p_focus.yu/H,p_focus.yd/H])
            think=birds[i].brain.query([birds[i].y/H,birds[i].vy/H,p_focus.x/W,p_focus.yu/H])
            if(think>0.5):
                birds[i].jump()
            birds[i].score+=1
            collision=detect_collision(birds[i],pillars[0])
            if(collision==True):
                new_dead_birds+=[birds[i]]
            
        ################
    
        ######display######
        for i in range(len(pillars)):
            delete_obj(canvas,P[i])
            P[i]=pillars[i].display(canvas)

        for i in range(len(birds)):
            delete_obj(canvas,B[i])
        B=[b.display(canvas) for b in birds if b not in new_dead_birds]
   
        delete_obj(canvas,S)
        if(len(birds)>0):
            S=score_display(canvas,birds[0])
    
        canvas.update()
        ################
        #####game logic#####
        for db in new_dead_birds:
            birds.remove(db)
            dead_birds+=[db]
        ################
    
    calculate_fitness(dead_birds)
    print("best score : "+str(dead_birds[-1].score-210))
    
    #best=dead_birds[birdsize-10:]
    #calculate_fitness(best)

    birds=[pickOne(dead_birds) for k in range(birdsize)]
    #birds=[pickOne(best) for k in range(birdsize)]
    
    canvas.delete("all")  #display
   

root.mainloop()   #display

      
