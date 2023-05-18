import time,psutil
#import time,tracemalloc,psutil,gc
from pympler import muppy,summary
#tracemalloc.start()
t0 = time.time()
snaps = []

def snapshots(tmpdir,RANK,n1=5,n2=10):
    #snaps.append(tracemalloc.take_snapshot())

    ls = muppy.get_objects()
    sum_ = summary.summarize(ls) 
    summary.print_(sum_)
    #ls = muppy.filter(ls,Type=dict)
    #nprint = 10
    #every = len(ls) // nprint
    #print(ls[::every])

    mem = psutil.virtual_memory()
    print(f'time={time.time()-t0},percent={mem.percent}')
    #for tup in ls[::every]:
    #    print(tup,muppy.get_referents(tup,level=2))
    #with open(tmpdir+f'RANK{RANK}.log','a') as f:
        #f.write('\n')

        #cnt1 = gc.get_count()
        #gc.collect()
        #cnt2 = gc.get_count()
        #f.write(f'time={times[-1]-t0},gc_count1={cnt1},gc_count2={cnt2}\n')

        #mem = psutil.virtual_memory()
        #f.write(f'time={time.time()-t0},percent={mem.percent}\n')

        #ls = muppy.get_objects()
        #ls = muppy.filter(ls,Type=tuple)
        #nprint = 10
        #every = len(ls) // nprint
        #for tup in ls[::every]:
        #    print(tup,muppy.get_referents(tup,level=2))
        #sum_ = summary.summarize(ls) 
        #summary.print_(sum_)
        
        #snapshot = snaps[-1].filter_traces((
        #	    tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        #	    tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
        #	    tracemalloc.Filter(False, "<unknown>"),
        #    ))

        #f.write(f"*** top {n1} stats grouped by filename ***\n")
        #stats = snapshot.statistics('filename') 
        #for s in stats[:n1]:
        #    f.write(f'{s}\n')

        #stats = snapshot.statistics("traceback")
        #largest = stats[0]
        #f.write(f"*** Trace for largest memory block - ({largest.count} blocks, {largest.size/1024} Kb) ***\n")
        #for l in largest.traceback.format():
        #    f.write(f'{l}\n')

        #if len(snaps)>1:
        #    stats = snaps[-1].compare_to(snaps[0], 'lineno')
        #    f.write(f"*** top {n2} stats ***\n")
        #    for s in stats[:n2]:
        #        f.write(f'{s}\n')

