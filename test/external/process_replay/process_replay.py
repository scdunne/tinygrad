#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, multiprocessing, os, json
from typing import List, Tuple
from urllib.request import Request
from tinygrad.codegen.kernel import Kernel, OptOps
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, getenv, tqdm, _cache_dir, CI
from tinygrad.ops import LazyOp


from typing import Union, Optional
import urllib.request
import pathlib, tempfile, hashlib
def fetch(url:Union[str,urllib.request.Request], name:Optional[Union[pathlib.Path, str]]=None, subdir:Optional[str]=None,
          allow_caching=not getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
  if isinstance(url, str) and url.startswith(("/", ".")): return pathlib.Path(url)
  if name is not None and (isinstance(name, pathlib.Path) or '/' in name): fp = pathlib.Path(name)
  else:
    name = name if name is not None else url.full_url if isinstance(url, urllib.request.Request) else hashlib.md5(url.encode('utf-8')).hexdigest()
    fp = pathlib.Path(_cache_dir) / "tinygrad" / "downloads" / (subdir or "") / name
  if not fp.is_file() or not allow_caching:
    with urllib.request.urlopen(url, timeout=10) as r:
      assert r.status == 200
      total_length = int(r.headers.get('content-length', 0))
      progress_bar = tqdm(total=total_length, unit='B', unit_scale=True, desc=f"{url}", disable=CI)
      (path := fp.parent).mkdir(parents=True, exist_ok=True)
      with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        while chunk := r.read(16384): progress_bar.update(f.write(chunk))
        f.close()
        progress_bar.update(close=True)
        if (file_size:=os.stat(f.name).st_size) < total_length: raise RuntimeError(f"fetch size incomplete, {file_size} < {total_length}")
        pathlib.Path(f.name).rename(fp)
  return fp

page_size = 100
table_name = f"process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}_{VERSION}"

def is_master():
  # return True
  r = os.getenv("GITHUB_REF") == "master"
  print(r)
  return r

def process_replay(offset:int):
  ASSERT_PROCESS_REPLAY = (k:="[run_process_replay]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)
  ASSERT_PARAMS = (k:="[assert_params]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{table_name}' LIMIT ? OFFSET ?", (page_size, offset))
  params: List[Tuple[LazyOp, List[OptOps]]] = []
  for row in cur.fetchall():
    ast, opts, applied_opts, name, compare_src, ctx = pickle.loads(row[0])
    if is_master(): params.append((ast, applied_opts))
    if ASSERT_PARAMS:
      """
      upstream_params = fetch()
      try: assert any(t == (ast, applied_opts) for t in upstream_params)
      except AssertionError as e:
        print("AST OR APPLIED_OPTS CHANGED")
        print(ast)
        print(applied_opts)
        raise e
      """
    with Context(**{k:v for k,v in ctx.items() if k in ContextVar._cache}):
      # try linearize
      try:
        k = Kernel(ast, opts=opts)
        for opt in applied_opts: k.apply_opt(opt)
        good_src = k.opts.render(name, k.linearize().uops)
      except Exception as e:
        print("FAILED TO RECREATE KERNEL")
        print(ast)
        print(applied_opts)
        print(e)
        if ASSERT_PROCESS_REPLAY: raise e
        continue
      # try compare
      try: assert compare_src == good_src
      except AssertionError as e:
        print("PROCESS REPLAY DETECTED CHANGE")
        print(ast)
        print(applied_opts)
        diff = list(difflib.unified_diff(good_src.splitlines(), compare_src.splitlines()))
        for line in diff:
          print(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
        if ASSERT_PROCESS_REPLAY: raise e
  if is_master(): pickle.dump(params, open("/tmp/process_replay.pkl", "wb"))
  conn.commit()
  cur.close()

if __name__ == "__main__":
  """
  conn = db_connection()
  cur = conn.cursor()
  row_count = cur.execute(f"select count(*) from '{table_name}'").fetchone()[0]
  conn.commit()
  cur.close()
  offsets = range(0, row_count, page_size)
  with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: list(tqdm(pool.imap(process_replay, offsets), total=len(offsets)))
  """
  run_id = "9925440225"
  req = Request(f"https://api.github.com/repos/{getenv('GITHUB_REPOSITORY', '')}/actions/runs/{run_id}/artifacts?name=process_replay_gpu.pkl")
  req.add_header("Authorization", f"Bearer {getenv('GITHUB_TOKEN', '')}")
  req.add_header("Accept", "application/vnd.github+json")
  req.add_header("X-GitHub-Api-Version", "2022-11-28")
  download_url = json.load(open(fetch(req), "r"))["artifacts"][0]["archive_download_url"]
  req = Request(download_url)
  req.add_header("Authorization", f"Bearer token {getenv('GITHUB_TOKEN', '')}")
  req.add_header("Accept", "application/vnd.github+json")
  req.add_header("X-GitHub-Api-Version", "2022-11-28")
  val = fetch(req)
  print(val)
