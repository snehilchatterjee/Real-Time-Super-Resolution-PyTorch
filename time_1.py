import time
import torch
import torchvision
from torch import nn
import gc
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_time(model,ip,batch,cuda,use_amp=False):
    if(cuda):
        model=model.to('cuda')
        ip=ip.to('cuda')
        print(f'Using cuda\n')

    start_time = 0
    end_time = 0

    # Avoiding init time
    '''
    we run some dummy examples through the network to do a ‘GPU warm-up.’ 
    This will automatically initialize the GPU and prevent it from going into power-saving mode 
    when we measure time. (https://deci.ai/blog/measure-inference-time-deep-neural-networks/)
    '''
    
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=use_amp):
            _ = model(ip) # GPU WARMUP
    
    with torch.inference_mode():
        start_time = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=use_amp):
            _ = model(ip*5)                       # Multiplied by 5 to make sure old results aren't cached and just returned
        torch.cuda.synchronize()
        end_time = time.perf_counter()
    
    duration = end_time - start_time
    
    # Print the duration in seconds
    print(f"Duration: {duration} seconds\nFPS: {batch/duration:.0f}")

    del ip
    del model
    del _
    gc.collect()
    if(cuda):
        torch.cuda.empty_cache()
        
from models import Generator
generator = Generator()

ip = torch.rand(5,3,480,256)
calculate_time(generator,ip,5,True,True)
