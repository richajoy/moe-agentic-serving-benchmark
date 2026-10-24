# nccl_bandwidth.py: all-reduce bus bandwidth and all-to-all algorithm bandwidth across the GPUs.
#   torchrun --nproc-per-node 8 scripts/analysis/nccl_bandwidth.py   (inside the serving container)
import os, torch, torch.distributed as dist, time
dist.init_process_group("nccl"); r=dist.get_rank(); w=dist.get_world_size(); torch.cuda.set_device(r)
for mb in [1,16,256,1024]:
    x=torch.ones(mb*2**20//2,dtype=torch.bfloat16,device="cuda")
    for _ in range(5): dist.all_reduce(x)
    torch.cuda.synchronize(); t=time.time(); n=20
    for _ in range(n): dist.all_reduce(x)
    torch.cuda.synchronize(); dt=(time.time()-t)/n
    bus=2*(w-1)/w*mb*2**20/dt/1e9
    if r==0: print(f"allreduce {mb}MB: {dt*1e6:.0f} us, busBW {bus:.0f} GB/s")
    y=torch.empty_like(x); 
    for _ in range(5): dist.all_to_all_single(y,x)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): dist.all_to_all_single(y,x)
    torch.cuda.synchronize(); dt=(time.time()-t)/n
    if r==0: print(f"alltoall {mb}MB: {dt*1e6:.0f} us, algBW {(w-1)/w*mb*2**20/dt/1e9:.0f} GB/s")
