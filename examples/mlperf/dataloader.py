import os, random, pickle, functools, itertools
from typing import List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv, prod, Timing, Context
from collections import deque
from multiprocessing import Queue, Process, shared_memory, connection, Lock, cpu_count, Pool

class MyQueue:
  def __init__(self, multiple_readers=True, multiple_writers=True):
    self._reader, self._writer = connection.Pipe(duplex=False)
    self._rlock = Lock() if multiple_readers else None
    self._wlock = Lock() if multiple_writers else None
  def get(self):
    if self._rlock: self._rlock.acquire()
    ret = pickle.loads(self._reader.recv_bytes())
    if self._rlock: self._rlock.release()
    return ret
  def put(self, obj):
    if self._wlock: self._wlock.acquire()
    self._writer.send_bytes(pickle.dumps(obj))
    if self._wlock: self._wlock.release()

def shuffled_indices(n, seed=None):
  rng = random.Random(seed)
  indices = {}
  for i in range(n-1, -1, -1):
    j = rng.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]

def loader_process(q_in, q_out, X:Tensor, seed):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  from extra.datasets.imagenet import center_crop, preprocess_train

  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, fn, val = _recv
      img = Image.open(fn)
      img = img.convert('RGB') if img.mode != "RGB" else img

      if val:
        # eval: 76.08%, load in 0m7.366s (0m5.301s with simd)
        # sudo apt-get install libjpeg-dev
        # CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
        img = center_crop(img)
        img = np.array(img)
      else:
        # reseed rng for determinism
        if seed is not None:
          np.random.seed(seed * 2 ** 20 + idx)
          random.seed(seed * 2 ** 20 + idx)
        img = preprocess_train(img)

      # broken out
      #img_tensor = Tensor(img.tobytes(), device='CPU')
      #storage_tensor = X[idx].contiguous().realize().lazydata.realized
      #storage_tensor._copyin(img_tensor.numpy())

      # faster
      X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()

      # ideal
      #X[idx].assign(img.tobytes())   # NOTE: this is slow!
      q_out.put(idx)
    q_out.put(None)

def batch_load_resnet(batch_size=64, val=False, shuffle=True, seed=None):
  from extra.datasets.imagenet import get_train_files, get_val_files
  files = get_val_files() if val else get_train_files()
  from extra.datasets.imagenet import get_imagenet_categories
  cir = get_imagenet_categories()
  BATCH_COUNT = min(32, len(files) // batch_size)

  gen = shuffled_indices(len(files), seed=seed) if shuffle else iter(range(len(files)))
  def enqueue_batch(num):
    for idx in range(num*batch_size, (num+1)*batch_size):
      fn = files[next(gen)]
      q_in.put((idx, fn, val))
      Y[idx] = cir[fn.split("/")[-2]]

  shutdown = False
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch():
    while 1:
      num = q_out.get()//batch_size
      gotten[num] += 1
      if gotten[num] == batch_size: break
    gotten[num] = 0
    return X[num*batch_size:(num+1)*batch_size], Y[num*batch_size:(num+1)*batch_size], Cookie(num)

  #q_in, q_out = MyQueue(multiple_writers=False), MyQueue(multiple_readers=False)
  q_in, q_out = Queue(), Queue()

  sz = (batch_size*BATCH_COUNT, 224, 224, 3)
  if os.path.exists("/dev/shm/resnet_X"): os.unlink("/dev/shm/resnet_X")
  shm = shared_memory.SharedMemory(name="resnet_X", create=True, size=prod(sz))
  procs = []

  try:
    # disk:shm is slower
    #X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:shm:{shm.name}")
    X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/resnet_X")
    Y = [None] * (batch_size*BATCH_COUNT)

    for _ in range(cpu_count()):
      p = Process(target=loader_process, args=(q_in, q_out, X, seed))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    # NOTE: this is batch aligned, last ones are ignored
    for _ in range(0, len(files)//batch_size): yield receive_batch()
  finally:
    shutdown = True
    # empty queues
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    # shutdown processes
    for p in procs: p.join()
    shm.close()
    shm.unlink()

@functools.lru_cache(maxsize=128)
def load_bert_file(fn:str) -> List[dict]:
  with open(fn, "rb") as f: data = pickle.load(f)
  return data

def process_batch_bert(data: List[dict]) -> dict[str, Tensor]:
  return {
    "input_ids": Tensor(np.concatenate([s["input_ids"] for s in data], axis=0), dtype=dtypes.float32),
    "input_mask": Tensor(np.concatenate([s["input_mask"] for s in data], axis=0), dtype=dtypes.default_float),
    "segment_ids": Tensor(np.concatenate([s["segment_ids"] for s in data], axis=0), dtype=dtypes.float32),
    "masked_lm_positions": Tensor(np.concatenate([s["masked_lm_positions"] for s in data], axis=0), dtype=dtypes.float32),
    "masked_lm_ids": Tensor(np.concatenate([s["masked_lm_ids"] for s in data], axis=0), dtype=dtypes.float32),
    "masked_lm_weights": Tensor(np.concatenate([s["masked_lm_weights"] for s in data], axis=0), dtype=dtypes.float32),
    "next_sentence_labels": Tensor(np.concatenate([s["next_sentence_labels"] for s in data], axis=0), dtype=dtypes.float32),
  }

def shuffle_parts(file_paths: List[str]) -> List[str]:
  parts = {}
  for f in file_paths:
    part = Path(f).stem.split('_')[0]
    if part not in parts: parts[part] = []
    parts[part].append(f)
  
  part_ids = list(parts.keys())
  random.shuffle(part_ids)

  shuffled_files = []
  for p in part_ids:
    parts[p].sort(key=lambda x: int(Path(x).stem.split('_')[1]))
    shuffled_files.extend(parts[p])
  return shuffled_files

def random_sample(data: List[str]):
  index = random.randint(0, len(data) - 1)
  selected_sample = data[index]
  return selected_sample, index

def load_datasample(file_and_offset:Tuple[str, int]) -> List[dict]:
  data = load_bert_file(file_and_offset[0])
  return data[file_and_offset[1]]

# Reference: https://github.com/mlcommons/training/blob/1c8a098ae3e70962a4f7422c0b0bd35ae639e357/language_model/tensorflow/bert/run_pretraining.py, Line 394
def batch_load_train_bert(BS:int):
  from extra.datasets.wikipedia import get_wiki_train_files
  files = shuffle_parts(get_wiki_train_files())
  dataset = []
  for f in files: 
    lists = [(f, o) for o in range(int(Path(f).stem.split("_")[3].split(".")[0]))]
    dataset.extend(lists)
  
  active_set = deque(dataset[:1000])
  remaining_set = deque(dataset[1000:])

  while dataset:
    blob = []
    for _ in range(BS):
      if active_set:
        index = random.randint(0, len(active_set) - 1)
        sample = active_set[index]
        active_set.remove(sample)
        blob.append(sample)
        if remaining_set:
            active_set.append(remaining_set.popleft())
    yield process_batch_bert([load_datasample(sample) for sample in blob])

# Reference: https://github.com/mlcommons/training/blob/1c8a098ae3e70962a4f7422c0b0bd35ae639e357/language_model/tensorflow/bert/run_pretraining.py, Line 416
def batch_load_val_bert(BS:int):
  from extra.datasets.wikipedia import get_wiki_val_files
  files = get_wiki_val_files()
  dataset = list(itertools.chain.from_iterable([load_bert_file(f) for f in files]))
  idx = 0
  while True:
    start_idx = (idx * BS) % len(dataset)
    end_idx = ((idx + 1) * BS) % len(dataset)
    if start_idx < end_idx:
        yield process_batch_bert(dataset[start_idx:end_idx])
    else:  # wrap around the end to the beginning of the dataset
        yield process_batch_bert(dataset[start_idx:] + dataset[:end_idx])
    idx += 1

if __name__ == "__main__":
  from extra.datasets.imagenet import get_train_files, get_val_files
  VAL = getenv("VAL", 1)
  files = get_val_files() if VAL else get_train_files()
  with tqdm(total=len(files)) as pbar:
    for x,y,c in batch_load_resnet(val=VAL):
      pbar.update(x.shape[0])
