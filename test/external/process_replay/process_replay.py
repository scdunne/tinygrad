#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, multiprocessing, os
from typing import List, Tuple
from tinygrad.codegen.kernel import Kernel, OptOps
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, fetch, getenv, tqdm
from tinygrad.ops import LazyOp

page_size = 100
table_name = f"process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}_{VERSION}"

def is_master():
  return True
  return os.getenv("GITHUB_REF") == "master"

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
      params.append((ast, applied_opts))
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
  conn = db_connection()
  cur = conn.cursor()
  row_count = cur.execute(f"select count(*) from '{table_name}'").fetchone()[0]
  conn.commit()
  cur.close()
  offsets = range(0, row_count, page_size)
  with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: list(tqdm(pool.imap(process_replay, offsets), total=len(offsets)))
