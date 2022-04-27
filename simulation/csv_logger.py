#%%
import pandas as pd
import multiprocessing as mp


#%%
class Writer:
    def __init__(self) -> None:
        manager = mp.Manager()
        self.train = manager.list()
        self.eval = manager.list()
        self.ti = 0
        self.ei = 0 
        
    
    def write(self, data):
        print('from logger: ', data)
        if data['mode'] == 'train':
            self.train.append(pd.DataFrame(data,index=[self.ti]))
            self.ti += 1
        else:
            self.eval.append(pd.DataFrame(data,index=[self.ei]))
            self.ei += 1

    def to_csv(self,path):
        data = pd.concat(self.train, ignore_index=True)
        data.to_csv(path.replace('.csv', '_train.csv'))
        data = pd.concat(self.eval, ignore_index=True)
        data.to_csv(path.replace('.csv', '_eval.csv'))