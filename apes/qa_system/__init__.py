from os import environ

environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,optimizer=None"
