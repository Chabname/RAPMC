from gensim.models.callbacks import CallbackAny2Vec



class LossLogger(CallbackAny2Vec):
    out = "results/log_"
    '''Output loss at each epoch'''
    def __init__(self,filename):
        self.epoch = 1
        self.loss = 0
        self.losses = []
        self.out += filename + ".txt"

    def on_epoch_begin(self, model):
        with open(self.out, 'a') as f:
            print(f'Epoch: {self.epoch}', end='\t',  file=f)


    def on_epoch_end(self, model):
        self.loss = model.get_latest_training_loss()
        self.losses.append(self.loss)
        with open(self.out, 'a') as f:
            print(f'  Loss: {self.loss}',  file=f)
        self.epoch += 1

