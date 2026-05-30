import os
import datetime


def gen_tag(**kwargs):
    '''
    Generate a unique tag or name based on the model components
    Could include time if needed to guarantee uniqueness of models
    '''
    model = kwargs.get('model', '')
    opt = kwargs.get('opt', '')
    learn = kwargs.get('learn', '')
    include_time = kwargs.get('include_time',False)
    
    name = model.__name__()+'_'+opt.__name__()+'_'+str(opt.opt.defaults['lr'])
    if learn.sched !=None:
        sched_name = '_'+learn.sched.__name__()+'_'+str(learn.sched.sched.eta_min)
        name += sched_name
        
    if include_time:
        name +=f'_{datetime.datetime.now()}'[:17]
        
    return name
    
def create_dir(tag):
    '''Creates directory in the given path and notify if dir already id writes in it'''
    newpath = tag 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print('best model is stored in ',newpath)
    
def save_all(tag,**kwargs):
    '''Save all model details (model,optimizer,scheduler)'''
    model = kwargs.get('model', '')
    opt = kwargs.get('opt', '')
    learn = kwargs.get('learn','')
    
    
    model.save(tag+model.__name__())
    opt.save(tag+opt.__name__())
    if learn.sched !=None:
        learn.sched.save(tag+learn.sched.__name__())
    

def load_all(path,**kwargs):
    '''Load all model details in place(model,optimizer,scheduler)'''
    model = kwargs.get('model', '')
    opt = kwargs.get('opt', '')
    learn = kwargs.get('learn','')
    
    path = 'Models/'+path
    model.load(path+model.__name__())
    opt.load(path+opt.__name__())
    if learn.sched !=None:
        learn.sched.load(path+learn.sched.__name__())
    
    