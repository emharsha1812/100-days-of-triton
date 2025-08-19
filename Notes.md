### Key terms

**Threads** - A thread is a simplified view of how a processor executes a sequential program in modern computers. It consists of the code of the program, the point in the code that is being executed and the values of its variables and data structures

The execution of a thread is sequential as far as a user is concerned

**Throughput** - Throughput is a measure of how much work is done in a specific amount of time. It’s all about efficiency and speed.In computing, "maximal throughput" means getting the most calculations or operations done per second. For a GPU (Graphics Processing Unit), this means making it run as fast as possible to process massive amounts of data.

**Kernel** - In GPU programming, a kernel is a small, specific function or piece of code that you want to run many, many times at the same time (in parallel).

In triton, kernels are defined as decorated Python functions, and launched concurrently with different program_id’s on a grid of so-called instances.

**JIT (Just in Time)** - Just-in-time (JIT) compilation is the process of compiling code at runtime into native machine instructions right before it’s needed

#### Advantages of JIT

1. Runs faster after the first time
2. Works on any machine
3. Gets better with new hardware or new runtime - No recompile needed

#### Where is this cached code part kept ?

In RAM while the program is open (so repeats in the same run are instant)

On your disk for next time (so next run also skips cooking):
NVIDIA GPUs: ~/.nv/ComputeCache by default.

    Triton kernels (used from PyTorch): a folder like ~/.triton/cache unless you change an env-var.

**IMP** - If you change the kernel’s source code or its fixed-size settings, Triton makes a new cached file the next time the kernel is called.

#### How is it different from Standard plain Cpython ?

1. When you run python myscript.py, the interpreter quickly turns the whole file into byte-code—a hardware-independent set of op-codes stored in a .pyc file inside the **pycache** folder.

2. The same process then steps through that byte-code one instruction at a time, a bit like reading a recipe card line-by-line.

These instructions are not native machine code, so each step still carries some overhead.

Because of step 1, the source isn’t re-parsed every line; it is first compiled once per run (or even skipped if an up-to-date .pyc already exists). But CPython never converts the code all the way down to the CPU’s real instructions the way a JIT does—so you don’t get the big speed jump that Triton, PyPy, Numba, etc. can provide.

**Summary** - CPython caches a quick “recipe card” (byte-code) and then interprets it line-by-line at run-time; a JIT goes a step further and bakes that recipe into raw machine code so future runs are much faster.

#### What is the difference between Byte-code and Machine code ?

## Byte-code vs. Machine Code — the essentials

| Aspect             | Byte-code                                                                                                                                                  | Machine code                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| What it is         | A **middle-man format** produced by a language runtime (Python, Java, .NET) after it parses your source code.                                              | The **final, native instructions** understood directly by a CPU or GPU.                                                 |
| Who runs it        | A **virtual machine or interpreter** that reads each byte-code step and decides what to do next.                                                           | The **hardware itself**; no extra interpreter layer.                                                                    |
| Portability        | Platform-independent—run the same `.pyc`/`.class` file anywhere the VM exists.                                                                             | Tied to one chip family (x86-64, ARM, NVIDIA SASS, etc.).                                                               |
| Speed              | Slower, because every instruction is still “looked up” by the VM at runtime. A JIT can speed it up by translating hot spots to machine code on the fly.[1] | Fastest possible: already in the CPU’s own language.                                                                    |
| What it looks like | Human-readable mnemonics plus small numeric codes, e.g. Python disassembly: `LOAD_FAST 0`, `BINARY_ADD`, `RETURN_VALUE`.                                   | Hex or binary, e.g. `48 8B 45 F8` (x86-64 “mov rax, [rbp-0x8]”). To people, it’s just streams of 0s, 1s, or hex digits. |

### Tiny visual peek

```python
# Python source --------------------------------
def add(a, b):
    return a + b
```

```text
# Python byte-code (dis.dis) --------------------
  0 LOAD_FAST                0 (a)
  2 LOAD_FAST                1 (b)
  4 BINARY_ADD
  6 RETURN_VALUE
```

```
# Equivalent x86-64 machine code ---------------
48 89 f8          ; mov    rax,rdi
48 01 f0          ; add    rax,rsi
c3                ; ret
```

_Byte-code_ still uses friendly op-codes like `LOAD_FAST`; _machine code_ has already collapsed to raw bytes that the processor decodes electrically.

### How they fit together in typical Python (CPython)

1. You run a `.py` file.
2. CPython quickly compiles it once to byte-code and saves that in `__pycache__/file.cpython-...pyc`.
3. The CPython interpreter then walks that byte-code one instruction at a time on every call.
4. Libraries such as **Numba**, **PyPy**, or GPU tools like **Triton** add a _JIT_: they notice hot byte-code sections, translate them further into machine code, and cache the result so future calls skip the interpreter altogether.

So:

- **Byte-code** = portable recipe cards.
- **Machine code** = the finished dish, ready to eat by the hardware—no waiter (interpreter) needed.
