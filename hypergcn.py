
# coding: utf-8

# In[ ]:


# coding: utf-8


# # parse arguments ([ConfigArgParse](https://github.com/bw2/ConfigArgParse))

# In[ ]:


from config import config
args = config.parse()


# # gpu

# In[ ]:


import os, torch, numpy as np
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)



# # load data

# In[ ]:


from data import data
dataset, train, test = data.load(args)
print("length of train is", len(train))





# # initialise HyperGCN

# In[ ]:


from model import model
HyperGCN = model.initialise(dataset, args)


# # train and test HyperGCN

# In[ ]:


HyperGCN = model.train(HyperGCN, dataset, train, args)
acc = model.test(HyperGCN, dataset, test, args)
print(float(acc))


# # store result

# In[ ]:



# #  'r': [run all cells](https://stackoverflow.com/questions/33143753/jupyter-ipython-notebooks-shortcut-for-run-all)

# In[ ]:


