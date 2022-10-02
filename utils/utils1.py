import numpy as np
import os
import datetime
import _settings
import tqdm

LOG_FOLDER = _settings.LOG_OUTPUT_DIR
TODAY_STR = datetime.datetime.today().strftime('%Y%m%d')


def merge_dict_inline(d1,d2):
    d1 = d1.copy()
    d1.update(d2)
    return d1

#========================
import  logging
def get_logger(name=None, log_path=None, level = logging.INFO):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if len(logger.handlers) > 0 and log_path is None: return logger
    if log_path is not None:
        if not log_path.endswith(".log"):
            log_path = os.path.join(log_path, "%s.log"%TODAY_STR)
        log_dir = os.path.dirname(log_path)
        if log_dir == '':
            log_dir = LOG_FOLDER
            log_path = os.path.join(log_dir, log_path)
    else:
        if name is None:
            log_path = os.path.join(LOG_FOLDER, "%s.log"%TODAY_STR)
        else:
            log_path = os.path.join(LOG_FOLDER, name, "%s.log"%TODAY_STR)
    log_dir = os.path.dirname(log_path)
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    if log_path is not None:
        for handler in [] if len(logger.handlers) == 0 else logger.handlers:
            if os.path.normpath(handler.baseFilename) == log_path:
                break
        else:
            logger.handlers = [] #TODO: This does not make sense in the current case. Maybe change the above to assuming there's only one handler
            fileHandler = logging.FileHandler(log_path, mode='a')
            fileHandler.setFormatter(logFormatter)
            logger.addHandler(fileHandler)
        #logger.warning("\n\nStart of a new log")
    return logger

class TaskPartitioner():
    def __init__(self, seed=None):
        self.task_list = None
        self.seed = seed or np.random.randint(100)

    def add_task(self, func, *args, **kwargs):
        if self.task_list is None:
            self.task_list = []
        else:
            assert isinstance(self.task_list, list), "Trying to add a task without key to a keyed TaskPartitioner"
        self.task_list.append((func, args, kwargs))


    def add_task_with_key(self, key, func, *args, **kwargs):
        if self.task_list is None:
            self.task_list = dict()
        else:
            assert isinstance(self.task_list, dict), "Trying to add a keyed task without key to a non-eyed TaskPartitioner"
        self.task_list[key] = (func, args, kwargs)

    def __len__(self):
        return len(self.task_list)

    def copy(self):
        import copy
        o = TaskPartitioner()
        o.task_list = copy.copy(self.task_list)
        return o

    def run(self, ith, shuffle=True, npartition=3, suppress_exception=False, cache_only=False, debug=False, process_kwarg=None):
        n = len(self.task_list)
        keyed = isinstance(self.task_list, dict)
        if ith is None:
            ith, npartition = 0, 1
        if shuffle:
            np.random.seed(npartition + self.seed)  # being lazy
            perm = np.random.permutation(len(self.task_list))
        else:
            perm= np.arange(n)
        if keyed:
            task_ids = [key for i, key in enumerate(self.task_list.keys()) if perm[i] % npartition == ith]
        else:
            task_ids = [perm[i] for i in range(n) if i % npartition == ith]
        res = {}
        for task_id in tqdm.tqdm(task_ids, ncols=int(_settings.NCOLS / 2 * 1.5)):
            func, arg, kwargs = self.task_list[task_id]
            if process_kwarg is not None: kwargs[process_kwarg] = ith
            if debug:
                print(func, arg, kwargs)
            try:
                res[task_id] = func(*arg, **kwargs)
                if cache_only: res[task_id] = True
            except Exception as err:
                if suppress_exception:
                    print(err)
                else:
                    raise err
        return res

    def run_multi_process(self, nprocesses=1, cache_only=True, process_kwarg=None):
        if self.task_list is None: return
        if nprocesses == 1: return self.run(None, shuffle=False, debug=False, process_kwarg=process_kwarg)
        if not cache_only: o2 = self.copy()
        from multiprocessing import Process
        ps = []
        for i in range(nprocesses):
            p = Process(target=self.run, args=(i,), kwargs={'npartition': nprocesses, 'suppress_exception': True, 'cache_only': True, 'process_kwarg': process_kwarg})
            p.start()
            ps.append(p)
        for i,p in enumerate(ps):
            p.join()
        if not cache_only:
            return o2.run(None, shuffle=False)


def set_all_seeds(random_seed=_settings.RANDOM_SEED, quiet=True):
    import torch
    import numpy as np
    # torch.set_deterministic(True)#This is only available in 1.7
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if not quiet: print("Setting seeds to %d" % random_seed)
    #os.environ["DEBUSSY"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def gpuid_to_device(gpuid, mod=True):
    if isinstance(gpuid, str):
        if gpuid.startswith('cpu') or gpuid.startswith('cuda'): return gpuid
        raise ValueError
    if isinstance(gpuid, list) or isinstance(gpuid, tuple):
        return tuple([gpuid_to_device(_gpuid) for _gpuid in gpuid])
    if gpuid == -1: return 'cpu'
    if gpuid is None: return 'cuda'
    if mod:
        import torch
        gpuid = gpuid % torch.cuda.device_count()
    return 'cuda:%d'%gpuid


#=====================================Hash for caching
import hashlib
import pickle
def _hash(k):
    return int(hashlib.md5(pickle.dumps(k, protocol=3)).hexdigest(), 16)
