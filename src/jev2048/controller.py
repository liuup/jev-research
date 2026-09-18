"""Closed-loop demonstrations are NOT calibrated policy evaluation under pi0."""
import numpy as np
from .evaluation import predict

class JevController:
    def __init__(self,model,tokenizer,utility="threshold",threshold=2048,max_length=512):
        self.model,self.tokenizer=model,tokenizer
        self.utility,self.threshold,self.max_length=utility,threshold,max_length

    def choose(self,game):
        actions=game.legal_actions
        rows=[dict(**game.state(),action=a,state_id=f"demo:{game.steps}") for a in actions]
        # All legal action questions and all outcome candidates share one backbone batch.
        p=predict(self.model,self.tokenizer,rows,len(rows),self.max_length)
        if self.utility=="threshold":
            if self.threshold not in (256,512,1024,2048,4096,8192): raise ValueError(self.threshold)
            values=np.array([128,256,512,1024,2048,4096,8192])
            utility=p[:,values>=self.threshold].sum(-1)
        elif self.utility=="log_tile": utility=p@np.array([7.,8.,9.,10.,11.,12.,13.])
        else: raise ValueError(self.utility)
        return actions[int(utility.argmax())]
