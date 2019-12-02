# DSCR
## Introduction 

Deep Social Collobrative Ranking (DSCR) is a new recommendation framework tailored to knowledge graph-based personalized recommendation. DSCR fully exploits both the social influence from potential friends and the collaborative influence from interactions for better embedding learning. 

## Environment Requirement
+ Python == 3.7
+ numpy == 1.16.2


## Datasets

+ Ciao & Epinion
   + In the Ciao and Epinion datasets, we have user' ratings towards items. The data is saved in a txt file (rating.txt) and the format is as follows:
   
         userid itemid rating
   
   + trust.txt: it contains the trust relations between users. There are two columns and both of them are userid, denoting there is a social relation between two users. 
   
         userid userid
         
   + train.txt: it contains data for train. Each line is a user with a list of her interacted items. 
   + test.txt: it contains data for test. Each line is a user with a list of her test items.
   
## Model 

+ Deep Social Collaborative Ranking (DSCR.py)

   + Model for Deep Social Collaborative Ranking

+ Running Command

         python3 DSCR.py --dataset ciao --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --layer_size_S [64,64] --lr 0.0005 --batch_size 1024 --epoch 400 
         
         python3 DSCR.py --dataset epinion --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --layer_size_S [64,64] --lr 0.0005 --batch_size 1024 --epoch 400 
   
   + You need to specify serveral parameters for training and testing:
   
    + dataset: ciao / epinion
    + regs: regularization weight 
    + layer_size: the number of layers and embedding size for user-item interaction network 
    + layer_size_s : the number of layers and embedding size for user-user social network
    + lr: learning rate
    + batch_size : the size of batch for training
    + epoch : the epoch for training 
   
   
   
   
