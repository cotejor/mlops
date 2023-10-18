import random

class Modelo2:
    __init__(self):
        pass

    def predict_proba(self,sample):
        return random.uniform(0,1)

deepgmi = Modelo2()
print(deepgmi.predict_proba({"Var1":4, "Var2":7}))


        