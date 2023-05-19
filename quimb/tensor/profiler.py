import time,tracemalloc,psutil,gc
#from pympler import muppy,summary
tracemalloc.start()
t0 = time.time()
snaps = {'curr':tracemalloc.take_snapshot()}

def snapshots(tmpdir,RANK,n1=5,n2=10):
    snaps['prev'] = snaps['curr']
    snaps['curr'] = tracemalloc.take_snapshot()
    with open(tmpdir+f'RANK{RANK}.log','a') as f:
        f.write('\n')

        cnt1 = gc.get_count()
        gc.collect()
        cnt2 = gc.get_count()

        mem = psutil.virtual_memory()
        f.write(f'time={time.time()-t0},percent={mem.percent},gc_count1={cnt1},gc_count2={cnt2}\n')

        snapshot = snaps['curr'].filter_traces((
        	    tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        	    tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
        	    tracemalloc.Filter(False, "<unknown>"),
            ))

        f.write(f"*** top {n1} stats grouped by filename ***\n")
        stats = snapshot.statistics('filename') 
        for s in stats[:n1]:
            f.write(f'{s}\n')

        stats = snapshot.statistics("traceback")
        largest = stats[0]
        f.write(f"*** Trace for largest memory block - ({largest.count} blocks, {largest.size/1024} Kb) ***\n")
        for l in largest.traceback.format():
            f.write(f'{l}\n')

        stats = snaps['curr'].compare_to(snaps['prev'], 'lineno')
        f.write(f"*** top {n2} stats ***\n")
        for s in stats[:n2]:
            f.write(f'{s}\n')

