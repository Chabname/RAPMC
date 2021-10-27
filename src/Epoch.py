from gensim.models.callbacks import CallbackAny2Vec



class LossLogger(CallbackAny2Vec):
    out = "results/log_"


    def __init__(self, filename):
        """ 
        Initialize loss loggger Object before completing

        function : 
           __init__
        input :
            filename : (str) Nale of the model used
        output :
            Object
        details : 
            Prepare the object
        """
        self.epoch = 1
        self.loss = 0
        self.losses = []
        self.out += filename + ".txt"


    def on_epoch_begin(self, model):
        """ 
        Write into the log file the number of epoch when the epoch begins

        function : 
           on_epoch_begin
        input :
            model : the model used
        output :
            log file updated
        """
        with open(self.out, 'a') as f:
            print(f'Epoch: {self.epoch}', end='\t',  file=f)


    def on_epoch_end(self, model):
        """ 
        Write into the log file loss value when the epoch ends

        function : 
           on_epoch_end
        input :
            model : the model used
        output :
            log file updated
        """
        self.loss = model.get_latest_training_loss()
        self.losses.append(self.loss)
        with open(self.out, 'a') as f:
            print(f'  Loss: {self.loss}',  file=f)
        self.epoch += 1

