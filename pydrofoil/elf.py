# =========================================================================
# elf
# =========================================================================
# A simple translator between ELF files and a sparse memory image object.
# Note that the translator is far from complete but is sufficient for use
# in our research processors. I found this document pretty useful for
# understanding the ELF32 format:
#
#  http://docs.oracle.com/cd/E19457-01/801-6737/801-6737.pdf
#
# Note that this implementation is inspired by the ELF object file reader
# here:
#
#  http://www.tinyos.net/tinyos-1.x/tools/src/mspgcc-pybsl/elf.py
#
# which includes this copyright:
#
#  (C) 2003 cliechti@gmx.net
#  Python license
#
# Author : Christopher Batten
# Date   : May 20, 2014
#
# BSD License
#
# adapted for pydrofoil by cfbolz

import struct

from rpython.rlib.rstruct.runpack import runpack as unpack
from rpython.rlib.rarithmetic import intmask

import binascii


class SparseMemoryImage(object):
    class Section(object):
        def __init__(self, name="", addr=0x00000000, data=bytearray()):
            self.name = name
            self.addr = addr
            self.data = data

        def __str__(self):
            return "{}: addr={} data={}".format(
                self.name, hex(self.addr), binascii.hexlify(self.data)
            )

        def __eq__(self, other):
            return (
                self.name == other.name
                and self.addr == other.addr
                and self.data == other.data
            )

    def __init__(self):
        self.sections = []
        self.symbols = {}

    def add_section(self, section, addr=None, data=None):
        if isinstance(section, SparseMemoryImage.Section):
            self.sections.append(section)
        else:
            sec = SparseMemoryImage.Section(section, addr, data)
            self.sections.append(sec)

    def get_section(self, section_name):
        for section in self.sections:
            if section_name == section.name:
                return section
        raise KeyError("Could not find section {}!", section_name)

    def get_sections(self):
        return self.sections

    def print_section_table(self):
        idx = 0
        print "Idx Name           Addr     Size"
        for section in self.sections:
            print(
                "{:>3} {:<14} {:0>8x} {}".format(
                    idx, section.name, section.addr, len(section.data)
                )
            )
            idx += 1

    def add_symbol(self, symbol_name, symbol_addr):
        self.symbols[symbol_name] = symbol_addr

    def get_symbol(self, symbol_name):
        return self.symbols[symbol_name]

    def __eq__(self, other):
        return self.sections == other.sections and self.symbols == other.symbols

    def __ne__(self, other):
        return not (self != other)

    def print_symbol_table(self):
        for key, value in self.symbols.iteritems():
            print(" {:0>8x} {}".format(value, key))


# -------------------------------------------------------------------------
# ELF File Format Types
# -------------------------------------------------------------------------
# These are the sizes for various ELF32 data types used in describing
# various structures below.
#
#            size alignment
# elf_addr   4    4  Unsigned program address
# elf_half   2    2  Unsigned medium integer
# elf_off    4    4  Unsigned file offset
# elf_sword  4    4  Signed   large integer
# elf_word   4    4  Unsigned large integer
# elf_byte   1    1  Unsigned small integer

# =========================================================================
# ElfHeader
# =========================================================================
# Class encapsulating an ELF32 header which implements the following
# C-structure.
#
# define EI_NIDENT 16
# typedef struct {
#   elf_byte e_ident[EI_NIDENT];
#   elf_half e_type;
#   elf_half e_machine;
#   elf_word e_version;
#   elf_addr e_entry;
#   elf_off  e_phoff;
#   elf_off  e_shoff;
#   elf_word e_flags;
#   elf_half e_ehsize;
#   elf_half e_phentsize;
#   elf_half e_phnum;
#   elf_half e_shentsize;
#   elf_half e_shnum;
#   elf_half e_shstrndx;
# } elf_ehdr;

# ELF64:
#
# unsigned char   e_ident [EI_NIDENT]
# Elf64_Half      e_type
# Elf64_Half      e_machine
# Elf64_Word      e_version
# Elf64_Addr      e_entry
# Elf64_Off       e_phoff
# Elf64_Off       e_shoff
# Elf64_Word      e_flags
# Elf64_Half      e_ehsize
# Elf64_Half      e_phentsize
# Elf64_Half      e_phnum
# Elf64_Half      e_shentsize
# Elf64_Half      e_shnum
# Elf64_Half      e_shstrndx


class ElfHeader(object):

    FORMAT64 = "<16sHHIQQQIHHHHHH"
    FORMAT = "<16sHHIIIIIHHHHHH"
    NBYTES = struct.calcsize(FORMAT)
    NBYTES64 = struct.calcsize(FORMAT64)

    # Offsets within e_ident

    IDENT_NBYTES = 16  # Size of e_ident[]
    IDENT_IDX_MAG0 = 0  # File identification
    IDENT_IDX_MAG1 = 1  # File identification
    IDENT_IDX_MAG2 = 2  # File identification
    IDENT_IDX_MAG3 = 3  # File identification
    IDENT_IDX_CLASS = 4  # File class
    IDENT_IDX_DATA = 5  # Data encoding
    IDENT_IDX_VERSION = 6  # File version
    IDENT_IDX_PAD = 7  # Start of padding bytes

    # Elf file type flags

    TYPE_NONE = 0  # No file type
    TYPE_REL = 1  # Relocatable file
    TYPE_EXEC = 2  # Executable file
    TYPE_DYN = 3  # Shared object file
    TYPE_CORE = 4  # Core file
    TYPE_LOPROC = 0xFF00  # Processor-specific
    TYPE_HIPROC = 0xFFFF  # Processor-specific

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    # def __init__( self, data=None ):
    #  if data != None:
    def __init__(self, data="", is_64bit=False):
        self.is_64bit = is_64bit
        if is_64bit:
            self.format = ElfHeader.FORMAT64
        else:
            self.format = ElfHeader.FORMAT

        if data != "":
            self.from_bytes(data)

    # -----------------------------------------------------------------------
    # from_bytes
    # -----------------------------------------------------------------------

    def from_bytes(self, data):
        ehdr_list = unpack(self.format, data)
        self.ident = ehdr_list[0]
        self.type = ehdr_list[1]
        self.machine = ehdr_list[2]
        self.version = ehdr_list[3]
        self.entry = ehdr_list[4]
        self.phoff = ehdr_list[5]
        self.shoff = ehdr_list[6]
        self.flags = ehdr_list[7]
        self.ehsize = ehdr_list[8]
        self.phentsize = ehdr_list[9]
        self.phnum = ehdr_list[10]
        self.shentsize = ehdr_list[11]
        self.shnum = ehdr_list[12]
        self.shstrndx = ehdr_list[13]

    # -----------------------------------------------------------------------
    # to_bytes
    # -----------------------------------------------------------------------

    def to_bytes(self):
        return struct.pack(
            ElfHeader.FORMAT,
            self.ident,
            self.type,
            self.machine,
            self.version,
            self.entry,
            self.phoff,
            self.shoff,
            self.flags,
            self.ehsize,
            self.phentsize,
            self.phnum,
            self.shentsize,
            self.shnum,
            self.shstrndx,
        )

    # -----------------------------------------------------------------------
    # __str__
    # -----------------------------------------------------------------------

    def __str__(self):
        return """
 ElfHeader:
   ident     = {},
   type      = {},
   machine   = {},
   version   = {},
   entry     = {},
   phoff     = {},
   shoff     = {},
   flags     = {},
   ehsize    = {},
   phentsize = {},
   phnum     = {},
   shentsize = {},
   shnum     = {},
   shstrndx  = {}
""".format(
            self.ident,
            self.type,
            self.machine,
            self.version,
            hex(self.entry),
            hex(self.phoff),
            hex(self.shoff),
            hex(self.flags),
            self.ehsize,
            self.phentsize,
            self.phnum,
            self.shentsize,
            self.shnum,
            self.shstrndx,
        )


# =========================================================================
# ElfSectionHeader
# =========================================================================
# Class encapsulating an ELF32 section header which implements the
# following C-structure.
#
# typedef struct {
#   elf_word sh_name;
#   elf_word sh_type;
#   elf_word sh_flags;
#   elf_addr sh_addr;
#   elf_off  sh_offset;
#   elf_word sh_size;
#   elf_word sh_link;
#   elf_word sh_info;
#   elf_word sh_addralign;
#   elf_word sh_entsize;
# } elf_shdr;
#

# ELF64:
#
# Elf64_Word  sh_name
# Elf64_Word  sh_type
# Elf64_Xword   sh_flags
# Elf64_Addr  sh_addr
# Elf64_Off   sh_offset
# Elf64_Xword   sh_size
# Elf64_Word  sh_link
# Elf64_Word  sh_info
# Elf64_Xword   sh_addralign
# Elf64_Xword   sh_entsize


class ElfSectionHeader(object):

    FORMAT = "<IIIIIIIIII"
    FORMAT64 = "<IIQQQQIIQQ"
    NBYTES = struct.calcsize(FORMAT)
    NBYTES64 = struct.calcsize(FORMAT64)

    # Section types. Note that we only load some of these sections.

    TYPE_NULL = 0
    TYPE_PROGBITS = 1  # \
    TYPE_SYMTAB = 2  # | We only load sections of these types
    TYPE_STRTAB = 3  # /
    TYPE_RELA = 4
    TYPE_HASH = 5
    TYPE_DYNAMIC = 6
    TYPE_NOTE = 7
    TYPE_NOBITS = 8
    TYPE_REL = 9
    TYPE_SHLIB = 10
    TYPE_DYNSYM = 11
    TYPE_LOPROC = 0x70000000
    TYPE_HIPROC = 0x7FFFFFFF
    TYPE_LOUSER = 0x80000000
    TYPE_HIUSER = 0xFFFFFFFF

    # Section attribute flags. Note that we only load sections with the
    # SHF_ALLOC flag set into the actual sparse memory.

    FLAGS_WRITE = 0x1
    FLAGS_ALLOC = 0x2
    FLAGS_EXECINSTR = 0x4
    FLAGS_MASKPROC = 0xF0000000

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, data="", is_64bit=False):
        self.is_64bit = is_64bit
        if is_64bit:
            self.format = ElfSectionHeader.FORMAT64
        else:
            self.format = ElfSectionHeader.FORMAT
        if data != "":
            self.from_bytes(data)

    # -----------------------------------------------------------------------
    # from_bytes
    # -----------------------------------------------------------------------

    def from_bytes(self, data):
        shdr_list = unpack(self.format, data)
        self.name = shdr_list[0]
        self.type = shdr_list[1]
        self.flags = shdr_list[2]
        self.addr = shdr_list[3]
        self.offset = shdr_list[4]
        self.size = shdr_list[5]
        self.link = shdr_list[6]
        self.info = shdr_list[7]
        self.addralign = shdr_list[8]
        self.entsize = shdr_list[9]

    # -----------------------------------------------------------------------
    # to_bytes
    # -----------------------------------------------------------------------

    def to_bytes(self):
        return struct.pack(
            ElfSectionHeader.FORMAT,
            self.name,
            self.type,
            self.flags,
            self.addr,
            self.offset,
            self.size,
            self.link,
            self.info,
            self.addralign,
            self.entsize,
        )

    # -----------------------------------------------------------------------
    # __str__
    # -----------------------------------------------------------------------

    def __str__(self):
        return """
 ElfSectionHeader:
   name      = {},
   type      = {},
   flags     = {},
   addr      = {},
   offset    = {},
   size      = {},
   link      = {},
   info      = {},
   addralign = {},
   entsize   = {},
""".format(
            self.name,
            self.type,
            hex(self.flags),
            hex(self.addr),
            hex(self.offset),
            self.size,
            self.link,
            self.info,
            self.addralign,
            self.entsize,
        )


# =========================================================================
# ElfSymTabEntry
# =========================================================================
# Class encapsulating an ELF32 symbol table entry which implements the
# following C-structure.
#
# typedef struct {
#   elf_word st_name;
#   elf_addr st_value;
#   elf_word st_size;
#   elf_byte st_info;
#   elf_byte st_other;
#   elf_half st_shndx;
# } elf_sym;
#


class ElfSymTabEntry(object):

    FORMAT = "<IIIBBH"
    NBYTES = struct.calcsize(FORMAT)

    # Symbol types. Note we only load some of these types.

    TYPE_NOTYPE = 0  # \
    TYPE_OBJECT = 1  # | We only load symbols of these types
    TYPE_FUNC = 2  # /
    TYPE_SECTION = 3
    TYPE_FILE = 4
    TYPE_LOPROC = 13
    TYPE_HIPROC = 15

    def __init__(self, data=""):
        if data != "":
            self.from_bytes(data)


    def from_bytes(self, data):
        sym_list = unpack(ElfSymTabEntry.FORMAT, data)
        self.name = sym_list[0]
        self.value = sym_list[1]
        self.size = sym_list[2]
        self.info = sym_list[3]
        self.other = sym_list[4]
        self.shndx = sym_list[5]

    def __str__(self):
        return """
 ElfSymTabEntry:
   ident     = {}
   value     = {}
   size      = {}
   info      = {}
   other     = {}
   shndx     = {}
""".format(
            self.name,
            hex(self.value),
            self.size,
            self.info,
            self.other,
            self.shndx,
        )


# -------------------------------------------------------------------------
# elf_reader
# -------------------------------------------------------------------------
# Opens and parses an ELF file into a sparse memory image object.


def elf_reader(file_obj, is_64bit=False):
    # Read the data for the ELF header
    ehdr_data = file_obj.read(ElfHeader.NBYTES64 if is_64bit else ElfHeader.NBYTES)
    ehdr = ElfHeader(ehdr_data, is_64bit=is_64bit)

    # Verify if its a known format and really an ELF file
    if ehdr.ident[0:4] != "\x7fELF":
        raise ValueError("Not a valid ELF file")

    # We need to find the section string table so we can figure out the
    # name of each section. We know that the section header for the section
    # string table is entry shstrndx, so we first get the data for this
    # section header.

    file_obj.seek(intmask(ehdr.shoff) + ehdr.shstrndx * ehdr.shentsize)
    shdr_data = file_obj.read(ehdr.shentsize)

    # Construct a section header object for the section string table

    shdr = ElfSectionHeader(shdr_data, is_64bit=is_64bit)

    # Read the data for the section header table

    file_obj.seek(intmask(shdr.offset))
    shstrtab_data = file_obj.read(intmask(shdr.size))

    # Load sections

    symtab_data = None
    strtab_data = None

    mem_image = SparseMemoryImage()

    for section_idx in range(ehdr.shnum):

        # Read the data for the section header

        file_obj.seek(intmask(ehdr.shoff) + section_idx * ehdr.shentsize)
        shdr_data = file_obj.read(ehdr.shentsize)

        # Pad the returned string in case the section header is not long
        # enough (otherwise the unpack function would not work)

        # shdr_data = shdr_data.ljust( ElfSectionHeader.NBYTES, '\0' )
        shdr_nbytes = ElfSectionHeader.NBYTES64 if is_64bit else ElfSectionHeader.NBYTES
        fill = "\0" * (shdr_nbytes - len(shdr_data))
        shdr_data = shdr_data + fill

        # Construct a section header object

        shdr = ElfSectionHeader(shdr_data, is_64bit=is_64bit)

        # Find the section name

        # start = shstrtab_data[shdr.name:]
        idx = shdr.name
        assert idx >= 0
        start = shstrtab_data[idx:]

        # section_name = start.partition('\0')[0]
        section_name = start.split("\0", 1)[0]

        # only sections marked as lloc should be written to memory

        if not (shdr.flags & ElfSectionHeader.FLAGS_ALLOC):
            continue

        # Read the section data if it exists

        if section_name not in [".sbss", ".bss"]:
            file_obj.seek(intmask(shdr.offset))
            data = file_obj.read(intmask(shdr.size))

        # NOTE: the .bss and .sbss sections don't actually contain any
        # data in the ELF.  These sections should be initialized to zero.
        # For more information see:
        #
        # - http://stackoverflow.com/questions/610682/bss-section-in-elf-file

        else:
            data = "\0" * shdr.size

        # Save the data holding the symbol string table

        if shdr.type == ElfSectionHeader.TYPE_STRTAB:
            strtab_data = data

        # Save the data holding the symbol table

        elif shdr.type == ElfSectionHeader.TYPE_SYMTAB:
            symtab_data = data

        # Otherwise create section and append it to our list of sections

        else:
            section = SparseMemoryImage.Section(section_name, shdr.addr, data)
            mem_image.add_section(section)

    # Load symbols. We skip the first symbol since it both "designates the
    # first entry in the table and serves as the undefined symbol index".
    # For now, I have commented this out, since we are not really using it.

    # num_symbols = len(symtab_data) / ElfSymTabEntry.NBYTES
    # for sym_idx in xrange(1,num_symbols):
    #
    #   # Read the data for a symbol table entry
    #
    #   start = sym_idx * ElfSymTabEntry.NBYTES
    #   sym_data = symtab_data[start:start+ElfSymTabEntry.NBYTES]
    #
    #   # Construct a symbol table entry
    #
    #   sym = ElfSymTabEntry( sym_data )
    #
    #   # Get the symbol type
    #
    #   sym_type  = sym.info & 0xf
    #
    #   # Check to see if symbol is one of the three types we want to load
    #
    #   valid_sym_types = \
    #   [
    #     ElfSymTabEntry.TYPE_NOTYPE,
    #     ElfSymTabEntry.TYPE_OBJECT,
    #     ElfSymTabEntry.TYPE_FUNC,
    #   ]
    #
    #   # Check to see if symbol is one of the three types we want to load
    #
    #   if sym_type not in valid_sym_types:
    #     continue
    #
    #   # Get the symbol name from the string table
    #
    #   start = strtab_data[sym.name:]
    #   name = start.partition('\0')[0]
    #
    #   # Add symbol to the sparse memory image
    #
    #   mem_image.add_symbol( name, sym.value )

    return mem_image
