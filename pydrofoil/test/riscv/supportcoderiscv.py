from pydrofoil.supportcode import *
from pydrofoil.bitvector import Integer
from pydrofoil import elf
from pydrofoil import mem as mem_mod

from rpython.rlib.nonconst import NonConstant
from rpython.rlib.objectmodel import we_are_translated, always_inline
from rpython.rlib.jit import JitDriver, promote
from rpython.rlib.rarithmetic import r_uint, intmask, ovfcheck
from rpython.rlib.rrandom import Random
from rpython.rlib import jit

import time


def write_mem(machine, addr, content): # write a single byte
    machine.g.mem.write(addr, 1, content)
    return True

def platform_read_mem(machine, executable_flag, read_kind, addr_size, addr, n):
    n = n.toint()
    assert n <= 8
    addr = addr.touint()
    res = machine.g.mem.read(addr, n, executable_flag)
    return bitvector.from_ruint(n*8, res)

def platform_write_mem(machine, write_kind, addr_size, addr, n, data):
    n = n.toint()
    assert n <= 8
    assert addr_size == 64
    assert data.size == n * 8
    machine.g.mem.write(addr.touint(), n, data.touint())
    return True

# rough memory layout:
# | rom | clint | .... | ram <htif inside> ram

@jit.not_in_trace
def _observe_addr_range(machine, pc, addr, width, ranges):
    index = _find_index(ranges, addr, width)
    machine.g._mem_addr_range_next = index

@jit.elidable
def _get_likely_addr_range(g, pc, ranges):
    # not really at all elidable, but it does not matter. the result is only
    # used to produce some guards
    return g._mem_addr_range_next

def _find_index(ranges, addr, width):
    for index, (start, stop) in enumerate(ranges):
        if start <= addr and addr + width < stop:
            return index
    return -1

def promote_addr_region(machine, addr, width, offset, executable_flag):
    g = jit.promote(machine.g)
    width = intmask(machine.word_width_bytes(width))
    addr = intmask(addr)
    jit.jit_debug("promote_addr_region", width, executable_flag, jit.isconstant(width))
    if not jit.we_are_jitted() or jit.isconstant(addr) or not jit.isconstant(width):
        return
    if executable_flag:
        return
    pc = machine.r.zPC
    _observe_addr_range(machine, pc, addr, width, g._mem_ranges)
    range_index = _get_likely_addr_range(g, pc, g._mem_ranges)
    if range_index < 0 or width > 8:
        return
    # the next line produces two guards
    if g._mem_ranges[range_index][0] <= addr and addr < g._mem_ranges[range_index][1] - width:
        if width == 8 and addr & ((r_uint(1)<<63) | 0b111) == 0:
            # it's aligned and the highest bit is not set. tell the jit that the
            # last three bits and the highest bit are zero. can be removed with
            # known bits analysis later
            #jit.record_exact_value(addr & 1, 0)
            #jit.record_exact_value(addr & 0b111, 0)
            #jit.record_exact_value((addr + width) & 0b111, 0)
            jit.record_exact_value(r_uint(addr) & (r_uint(1)<<63), 0)
            #jit.record_exact_value((r_uint(addr) >> 1) & 1, 0)
            #jit.record_exact_value((r_uint(addr) >> 2) & 1, 0)
    return

class Globals(object):
    _immutable_fields_ = [
        'config_print_platform?', 'config_print_mem_access?',
        'config_print_reg?', 'config_print_instr?', 'config_print_rvfi?',
        'rv_clint_base?', 'rv_clint_size?', 'rv_htif_tohost?',
        'rv_rom_base?', 'rv_rom_size?', 'mem?',
        'rv_insns_per_tick?',
        '_mem_ranges?[*]',
        'rv64'
    ]

    def __init__(self, rv64=True):
        self.rv64 = rv64
        self._mem_addr_range_next = -1
        self.mem = None
        self.rv_enable_pmp                  = False
        self.rv_enable_zfinx                = False
        self.rv_enable_rvc                  = True
        self.rv_enable_next                 = False
        self.rv_enable_writable_misa        = True
        self.rv_enable_fdext                = True
        self.rv_enable_dirty_update         = False
        self.rv_enable_misaligned           = False
        self.rv_mtval_has_illegal_inst_bits = False

        self.rv_ram_base = r_uint(0x80000000)
        self.rv_ram_size = r_uint(0x4000000)

        self.rv_rom_base = r_uint(0x1000)
        self.rv_rom_size = r_uint(0x100)

        self.random = Random(1)

        self.rv_clint_base = r_uint(0x2000000)
        self.rv_clint_size = r_uint(0xc0000)

        self.rv_htif_tohost = r_uint(0x80001000)
        self.rv_insns_per_tick = 100

        self.dtb = None

        self.term_fd = 1

        self.reservation = r_uint(0)
        self.reservation_valid = False

        self.dump_dict = None

        self.config_print_instr = True
        self.config_print_reg = True
        self.config_print_mem_access = True
        self.config_print_platform = True
        self.config_print_rvfi = False

    def _init_ranges(self):
        self._mem_ranges = [
            (intmask(self.rv_rom_base), intmask(self.rv_rom_base + self.rv_rom_size)),
            (intmask(self.rv_clint_base), intmask(self.rv_clint_base + self.rv_clint_size)),
            (intmask(self.rv_ram_base), intmask(self.rv_htif_tohost)),
            (intmask(self.rv_htif_tohost), intmask(self.rv_htif_tohost + 16)),
            (intmask(self.rv_htif_tohost + 16), intmask(self.rv_ram_base + self.rv_ram_size)),
        ]
        for a, b in self._mem_ranges:
            assert b >= 8


DEFAULT_RSTVEC = 0x00001000

def rv_16_random_bits(machine):
    # pseudo-random for determinism for now
    return r_uint(machine.g.random.genrand32()) & r_uint(0xffff)


# C externs

def sys_enable_rvc(machine, _):
    return machine.g.rv_enable_rvc

def sys_enable_next(machine, _):
    return machine.g.rv_enable_next

def sys_enable_fdext(machine, _):
    return machine.g.rv_enable_fdext

def sys_enable_zfinx(machine, _):
    return machine.g.rv_enable_zfinx

def sys_enable_writable_misa(machine, _):
    return machine.g.rv_enable_writable_misa

def plat_enable_dirty_update(machine, _):
    return machine.g.rv_enable_dirty_update

def plat_enable_misaligned_access(machine, _):
    return machine.g.rv_enable_misaligned

def plat_mtval_has_illegal_inst_bits(machine, _):
    return machine.g.rv_mtval_has_illegal_inst_bits

def plat_enable_pmp(machine, _):
    return machine.g.rv_enable_pmp

def plat_ram_base(machine, _):
    return machine.g.rv_ram_base

def plat_ram_size(machine, _):
    return machine.g.rv_ram_size

def plat_rom_base(machine, _):
    return machine.g.rv_rom_base

def plat_rom_size(machine, _):
    return machine.g.rv_rom_size

# Provides entropy for the scalar cryptography extension.
def plat_get_16_random_bits(machine, _):
    return rv_16_random_bits(machine)

def plat_clint_base(machine, _):
    return machine.g.rv_clint_base

def plat_clint_size(machine, _):
    return machine.g.rv_clint_size


def load_reservation(machine, addr):
    machine.g.reservation = addr
    machine.g.reservation_valid = True
    #print "reservation <- 0x%x" % (addr, )
    return ()

def speculate_conditional(machine, _):
    return True

def check_mask(machine):
    return r_uint(0x00000000FFFFFFFF) if machine.l.zxlen_val == 32 else r_uint(0xffffffffffffffff)

def match_reservation(machine, addr):
    mask = check_mask(machine)
    ret = machine.g.reservation_valid and (machine.g.reservation & mask) == (addr & mask)
    return ret

def cancel_reservation(machine, _):
    machine.g.reservation_valid = False
    return ()

def plat_term_write(machine, s):
    import os
    os.write(machine.g.term_fd, chr(s & 0xff))
    return ()

def plat_insns_per_tick(machine, _):
    pass

def plat_htif_tohost(machine, _):
    return machine.g.rv_htif_tohost

def memea(len, n):
    return ()


# sim stuff

def plat_term_write_impl(c):
    os.write(1, c)

def init_sail(machine, elf_entry):
    machine.init_model()
    init_sail_reset_vector(machine, elf_entry)
    if not machine.g.rv_enable_rvc:
        # this is probably unnecessary now; remove
        machine.set_Misa_C(machine.r.zmisa, 0)

def is_32bit_model(machine):
    return not machine.g.rv64

def init_sail_reset_vector(machine, entry):
    RST_VEC_SIZE = 8
    reset_vec = [ # 32 bit entries
        r_uint(0x297),                                      # auipc  t0,0x0
        r_uint(0x28593 + (RST_VEC_SIZE * 4 << 20)),         # addi   a1, t0, &dtb
        r_uint(0xf1402573),                                 # csrr   a0, mhartid
        r_uint(0x0182a283)  # lw     t0,24(t0)
        if is_32bit_model(machine) else
        r_uint(0x0182b283), # ld     t0,24(t0)
        r_uint(0x28067),                                    # jr     t0
        r_uint(0),
        r_uint(entry & 0xffffffff),
        r_uint(entry >> 32),
    ]


    rv_rom_base = DEFAULT_RSTVEC
    addr = r_uint(rv_rom_base)
    for i, fourbytes in enumerate(reset_vec):
        for j in range(4):
            write_mem(machine, addr, fourbytes & 0xff) # little endian
            addr += 1
            fourbytes >>= 8
        assert fourbytes == 0
    if machine.g.dtb:
        for i, char in enumerate(machine.g.dtb):
            write_mem(machine, addr, r_uint(ord(char)))
            addr += 1

    align = 0x1000
    # zero-fill to page boundary
    rom_end = r_uint((addr + align - 1) / align * align)
    for i in range(intmask(addr), rom_end):
        write_mem(machine, addr, 0)
        addr += 1

    # set rom size
    rv_rom_size = rom_end - rv_rom_base
    if machine.g.rv_rom_size != rv_rom_size:
        machine.g.rv_rom_size = rv_rom_size

    # boot at reset vector
    machine.r.zPC = r_uint(rv_rom_base)

def parse_dump_file(fn):
    with open(fn) as f:
        content = f.read()
    dump = {}
    section = '?'
    function = '?'
    for line in content.splitlines(False):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Disassembly of section "):
            section = line[len("Disassembly of section "):]
        elif line.endswith(">:"):
            pos = line.rfind("<")
            if pos < 0:
                continue
            endpos = len(line) - 2
            assert endpos >= 0
            function = line[pos + 1: endpos]
        else:
            pos = line.find(":")
            if pos < 0:
                continue
            address = line[:pos]
            res = line[pos + 1:].strip()
            intaddress = r_uint(0)
            for c in address:
                charval = -1
                if '0' <= c <= '9':
                    charval = ord(c) - ord('0')
                elif 'a' <= c <= 'f':
                    charval = ord(c) - ord('a') + 10
                elif 'A' <= c <= 'F':
                    charval = ord(c) - ord('A') + 10
                else:
                    break
                intaddress = intaddress * r_uint(16) + r_uint(charval)
            else:
                dump[intaddress] = "%s %s %s" % (section, function, res)
    return dump

def parse_flag(argv, flagname):
    return bool(parse_args(argv, flagname, want_arg=False))

helptext = """
Usage: %s [options] <elf_file>
-b/--device-tree-blob <file>    load dtb from file
-l/--inst-limit <limit>         exit after limit instructions have been executed
--dump <file>                   load elf file disassembly from file
--instructions-per-tick <num>   tick the emulated clock every num instructions
--verbose                       print a detailed trace of every instruction executed
--print-kips                    print kip/s every 2**20 instructions
--jit <options>                 set JIT options
--help                          print this information and exit
"""

def print_help(program):
    print helptext % program

JIT_HELP = ["Advanced JIT options:", '', '']
JIT_HELP.extend([" %s=<value>\n     %s (default: %s)\n" % (
    key, jit.PARAMETER_DOCS[key], value)
    for key, value in jit.PARAMETERS.items()]
)
JIT_HELP.extend([" off", "    turn JIT off", "", " help", "    print this page"])
JIT_HELP = "\n".join(JIT_HELP)

def print_help_jit():
    print JIT_HELP

def main(machinecls, argv):
    from rpython.rlib import jit

    if parse_flag(argv, "--help"):
        print_help(argv[0])
        return 0

    blob = parse_args(argv, "-b", "--device-tree-blob")
    limit = 0
    str_limit = parse_args(argv, "-l", "--inst-limit")
    if str_limit:
        limit = int(str_limit)
        print "instruction limit", limit

    dump_file = parse_args(argv, "--dump")
    per_tick = parse_args(argv, "--instructions-per-tick")

    jitopts = parse_args(argv, "--jit")
    if jitopts:
        if jitopts == "help":
            print_help_jit()
            return 0
        try:
            jit.set_user_param(None, jitopts)
        except ValueError:
            print "invalid jit option"
            return 1

    verbose = parse_flag(argv, "--verbose")

    print_kips = parse_flag(argv, "--print-kips")

    # Initialize model so that we can check or report its architecture.
    if len(argv) == 1:
        print_help(argv[0])
        return 1
    file = argv[1]
    if len(argv) == 3:
        iterations = int(argv[2])
    else:
        iterations = 1
    #init_logs()

    machine = machinecls()
    if blob:
        with open(blob, "rb") as f:
            machine.g.dtb = f.read()
    entry = load_sail(machine, file)
    init_sail(machine, entry)
    if not verbose:
        machine.g.config_print_instr = False
        machine.g.config_print_reg = False
        machine.g.config_print_mem_access = False
        machine.g.config_print_platform = False
    if dump_file:
        print "dump file", dump_file
        machine.g.dump_dict = parse_dump_file(dump_file)
    if per_tick:
        ipt = int(per_tick)
        machine.g.rv_insns_per_tick = ipt


    for i in range(iterations):
        run_sail(machine, limit, print_kips)
        if i:
            init_sail(machine, entry)
    #flush_logs()
    #close_logs()
    return 0

def get_printable_location(pc, do_show_times, insn_limit, tick, g):
    if tick:
        return "TICK 0x%x" % (pc, )
    if g.dump_dict and pc in g.dump_dict:
        return "0x%x: %s" % (pc, g.dump_dict[pc])
    return hex(pc)

driver = JitDriver(
    get_printable_location=get_printable_location,
    greens=['pc', 'do_show_times', 'insn_limit', 'tick', 'g'],
    reds=['step_no', 'insn_cnt', 'r', 'machine'],
    virtualizables=['r'])


def run_sail(machine, insn_limit, do_show_times):
    r = machine.r
    step_no = 0
    insn_cnt = 0
    tick = False

    machine.g._init_ranges()

    machine.g.interval_start = machine.g.total_start = time.time()
    prev_pc = 0

    while not r.zhtif_done and (insn_limit == 0 or step_no < insn_limit):
        driver.jit_merge_point(pc=r.zPC, tick=tick,
                insn_limit=insn_limit, step_no=step_no, insn_cnt=insn_cnt, r=r,
                do_show_times=do_show_times, machine=machine, g=machine.g)
        if tick:
            if insn_cnt == machine.g.rv_insns_per_tick:
                insn_cnt = 0
                machine.tick_clock()
                machine.tick_platform()
            else:
                assert do_show_times and (step_no & 0xfffff) == 0
                curr = time.time()
                print "kips:", 0x100000 / 1000. / (curr - machine.g.interval_start)
                machine.g.interval_start = curr
            tick = False
            continue
        # run a Sail step
        prev_pc = r.zPC
        stepped = machine.step(Integer.fromint(step_no))
        if r.have_exception:
            print "ended with exception!"
            print r.current_exception
            print "from", r.throw_location
            raise ValueError
        rv_insns_per_tick = machine.g.rv_insns_per_tick
        if stepped:
            step_no += 1
            if rv_insns_per_tick:
                insn_cnt += 1
        if machine.g.config_print_instr:
            # there's an extra newline in the C emulator that I don't know
            # where from, add it here to ease diffing
            print

        tick_cond = (do_show_times and (step_no & 0xffffffff) == 0) | (
                rv_insns_per_tick and insn_cnt == rv_insns_per_tick)
        if tick_cond:
            tick = True
        elif prev_pc >= r.zPC: # backward jump
            driver.can_enter_jit(pc=r.zPC, tick=tick,
                    insn_limit=insn_limit, step_no=step_no, insn_cnt=insn_cnt, r=r,
                    do_show_times=do_show_times, machine=machine, g=machine.g)
    # loop end

    interval_end = time.time()
    if r.zhtif_exit_code == 0:
        print "SUCCESS"
    else:
        print "FAILURE", r.zhtif_exit_code
        if not we_are_translated():
            raise ValueError
    print "Instructions: %s" % (step_no, )
    print "Total time (s): %s" % (interval_end - machine.g.total_start)
    print "Perf: %s Kips" % (step_no / 1000. / (interval_end - machine.g.total_start), )



def load_sail(machine, fn):
    g = machine.g
    oldmem = g.mem
    if oldmem:
        oldmem.close()
    mem1 = mem_mod.FlatMemory(False)
    mem2 = mem_mod.FlatMemory(False, g.rv_ram_size)
    mem = mem_mod.SplitMemory(mem1, 0, mem1.size, mem2, g.rv_ram_base, g.rv_ram_size)
    g.mem = mem
    with open(fn, "rb") as f:
        entrypoint = elf.elf_read_process_image(mem, f) # load process image
    with open(fn, "rb") as f:
        img = elf.elf_reader(f) # for the symbols

    g.rv_htif_tohost = r_uint(img.get_symbol('tohost'))
    print "tohost located at 0x%x" % g.rv_htif_tohost

    print "entrypoint 0x%x" % entrypoint
    assert entrypoint == 0x80000000 # XXX for now
    return entrypoint

# printing


def print_string(prefix, msg):
    print prefix, msg
    return ()

def print_instr(machine, s):
    print s
    return ()

print_reg = print_instr
print_mem_access = print_reg
print_platform = print_reg

def get_config_print_instr(machine, _):
    return machine.g.config_print_instr
def get_config_print_reg(machine, _):
    return machine.g.config_print_reg
def get_config_print_mem(machine, _):
    return machine.g.config_print_mem_access
def get_config_print_platform(machine, _):
    return machine.g.config_print_platform

def get_main(outriscv, rv64):
    class Machine(outriscv.Machine):
        _immutable_fields_ = ['g']
        def __init__(self):
            outriscv.Machine.__init__(self)
            self.g = Globals(rv64=rv64)

        def tick_clock(self):
            return outriscv.func_ztick_clock(self, ())

        def tick_platform(self):
            return outriscv.func_ztick_platform(self, ())

        def word_width_bytes(self, width):
            return outriscv.func_zword_width_bytes(self, width)

        def init_model(self):
            return outriscv.func_zinit_model(self, ())

        def set_Misa_C(self, *args):
            return outriscv.func_z_set_Misa_C(self, *args)

        def step(self, *args):
            return outriscv.func_zstep(self, *args)

    def bound_main(argv):
        return main(Machine, argv)
    from rpython.rlib import jit

    outriscv.Registers._virtualizable_ = ['ztlb39', 'ztlb48', 'zminstret', 'zPC', 'znextPC', 'zmstatus', 'zmip', 'zmie', 'zsatp']
    return bound_main