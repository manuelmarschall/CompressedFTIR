import multiprocessing as mp
from compressedftir.utils import stop_at_exception

def ex_fun(fun, queue, *args, **kwargs):
    """
    wraps a given function to allow exception handling when started in a new thread
    TODO: This is a perfect example for a decorator function

    Arguments:
        fun {callable} -- function to call
        queue {[type]} -- [description]
    """
    try:
        fun(*args, **kwargs)
        queue.put({"success": True})
    except Exception as ex:
        queue.put({"success": False})
        stop_at_exception(ex, "Multiprocessing has terminated!")    
    
def wrap_mp(fun, *args, **kwargs):
    """
    Wraps a given function and runs it in a multiprocessing environment.
    Fixes issues when plots are created in a loop.
    passing a multithreading queue to the process and checking the "success"
    variable allows for termination passing from subprocess exits.

    TODO: Again, decorator to remove the ugly call wrap_mp(fun, ...)
    TODO: Allow for return values passed by the queue
    TODO: Allow actual threading

    Arguments:
        fun {callable} -- function to call
    """
    mpqueue = mp.Queue()
    p = mp.Process(target=ex_fun, args=(fun, mpqueue, *args), kwargs=kwargs)
    p.start()
    res = mpqueue.get()
    if not res["success"]:
        exit()
    p.join()