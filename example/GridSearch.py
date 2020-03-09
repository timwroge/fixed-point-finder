from Utils import print_status
class GridSearch(object):
    def __init__(self, model_class, hyperparameters):
        '''The model is the callable that takes the hyper paramters  '''
        print_status ( ' Generating grid search with model: \n\t{} \nand hyperparameters: \n\t{} '.format(model_class, hyperparameters) )
        self.test_model_for_methods(model_class)
        self.hyperparameters = hyperparameters
        self.model_class = model_class

        self.param_set = {}
        # initialize the parameters
        for key in self.hyperparameters:
            if not isinstance(self.hyperparameters[key], list):
                raise Exception("All hyper parameters must be lists and {} is not a list".format(key))
            self.param_set[key] = {
                    "params" : self.hyperparameters[key],
                    "index" :  0,
                    "length": len(self.hyperparameters[key])}
        self.is_first = True

    def test_model_for_methods(self, model):
        '''This tests the model for the methods that are called during'''
        method = ""
        try:
            print_status(  "Testing model for method attributes" )
            method = 'train'
            assert hasattr(model, method)
            method = 'plot_trials'
            assert hasattr(model, method)
            method = 'predict'
            assert hasattr(model, method)
            method = '__init__'
            assert hasattr(model, method)
            print_status(  "The model succeeded model method tests" )
        except Exception:
            raise Exception("The model that was passed in does not have the method: ", method)
    def generate_model(self, hyperparameters):
        return self.model_class(**hyperparameters)
    def get_params(self):
        params = {}
        for param in self.param_set:
            params_array  = self.param_set[param]["params"]
            params_index  = self.param_set[param]["index"]
            params[param]  = params_array[params_index]
        return params


    def get_next_param_set(self):
        if (self.is_first):
            self.is_first = False
            return self.get_params()
        else:
            for param in self.param_set:
                #if any of them are not at their last index
                if param["index"] < param["length"]-1:
                    self.param_set[param]["index"] += 1
                    break
            #update the model params
            return self.get_params()

    def has_next_param_set(self):
        for param in self.param_set:
            #if any of them are not at their last index
            if self.param_set[param] [ "index"] < self.param_set[param]["length"]-1:
                return True
        return False

    def print_model_params(self, params):
        print_status( "Model Params:")
        for param in params:
            print "\t{}: {} ".format(param, params[param])


    def run(self):
        while self.has_next_param_set():
            params = self.get_next_param_set()
            self.print_model_params(params)
            model = self.generate_model(params)
            model.train()
            model.plot_trials()

