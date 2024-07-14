#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, multiprocessing, os
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, getenv, tqdm

page_size = 100
table_name = f"process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}_{VERSION}"

def process_replay(offset:int):
  ASSERT_PROCESS_REPLAY = (k:="[run_process_replay]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)
  ASSERT_PARAMS = (k:="[assert_params]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{table_name}' LIMIT ? OFFSET ?", (page_size, offset))
  for row in cur.fetchall():
    ast, opts, applied_opts, name, compare_src, ctx = pickle.loads(row[0])
    if ASSERT_PARAMS:
      upstream_params = pickle.load(open("/tmp/process_replay.pkl", "rb"))
      try: assert any(t == (ast, applied_opts) for t in upstream_params)
      except AssertionError as e:
        print("AST OR APPLIED_OPTS CHANGED")
        print(ast)
        print(applied_opts)
        raise e
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
  if getenv("PREPARE_PROCESS_REPLAY_UPLOAD"):
    conn = db_connection()
    cur = conn.cursor()
    drop_tables = cur.execute(f"SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table' and name != '{table_name}';").fetchall()
    cur.executescript("\n".join([s[0] for s in drop_tables]))
    conn.commit()
    cur.close()
