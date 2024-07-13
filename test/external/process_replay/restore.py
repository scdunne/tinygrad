import pickle
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import Device
from tinygrad.engine.graph import print_tree
from tinygrad.helpers import Context, db_connection, VERSION, ContextVar

conn = db_connection()
sha = "03c3b14cc23930eb86d55331f13b2d7135a8e836"
table_name = f"process_replay_{sha}_{VERSION}"

page_size = 100

if __name__ == "__main__":
  conn = db_connection()
  cur = conn.cursor()
  row_count = cur.execute(f"select count(*) from '{table_name}'").fetchone()[0]
  for offset in range(0, row_count, page_size):
    cur.execute(f"SELECT val FROM '{table_name}' LIMIT ? OFFSET ?", (page_size, offset))
    for row in cur.fetchall():
      ast, opts, applied_opts, name, compare_src, ctx = pickle.loads(row[0])
      with Context(**{k:v for k,v in ctx.items() if k in ContextVar._cache}):
        k = Kernel(ast, opts=opts)
        for opt in applied_opts: k.apply_opt(opt)
        src = k.opts.render(name, k.linearize().uops)
        try: Device[k.opts.device].compiler.compile(src)
        except Exception as e:
          print("FAILED TO COMPILE")
          for op in ast: print_tree(op)
          print(applied_opts)
          raise e
