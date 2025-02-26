from pydrofoil.parse import *

import os

thisdir = os.path.dirname(__file__)
cir = os.path.join(thisdir, "nand2tetris/generated/nand2tetris.jib")
riscvir = os.path.join(thisdir, "riscv/riscv_model_RV64.ir")

def test_lex_full():
    with open(cir, "rb") as f:
        s = f.read()
    try:
        for tok in lexer.lex(s):
            pass
    except LexingError as e:
        print s[e.getsourcepos().idx:e.getsourcepos().idx+20]
        assert 0

def test_lex_binary():
    tok, = lexer.lex("0b0100101")

def test_parse_enum():
    res = parser.parse(lexer.lex("""
enum zjump {
  zJDONT,
  zJGT,
  zJEQ,
  zJGE,
  zJLT,
  zJNE,
  zJLE,
  zJMP
}
"""))
    assert res.declarations[0] == Enum(
        'zjump', ['zJDONT', 'zJGT', 'zJEQ', 'zJGE', 'zJLT', 'zJNE', 'zJLE', 'zJMP'])

def test_union():
    res = parser.parse(lexer.lex("""
union zinstr {
  zAINST: %bv16,
  zCINST: (%bv1, %enum zarithmetic_op, (%bool, %bool, %bool), %enum zjump)
}
"""))
    assert res.declarations[0] == Union('zinstr',
            ['zAINST', 'zCINST'],
            [NamedType('%bv16'),
             TupleType([
                 NamedType('%bv1'), EnumType('zarithmetic_op'),
                 TupleType([NamedType('%bool'), NamedType('%bool'), NamedType('%bool')]),
                 EnumType('zjump')]),
            ])

def test_struct():
    res = parser.parse(lexer.lex("""
struct zXContextReg {
  zXContextReg_chunk_0: %bv64
}
"""))
    assert res.declarations[0] == Struct('zXContextReg',
        ['zXContextReg_chunk_0'],
        [NamedType('%bv64')],
    )

def test_let():
    res = parser.parse(lexer.lex("""
let (ztrace: %bool) {
  zgsz30_lz30 : %bool;
  zgsz30_lz30 = false;
  ztrace = zgsz30_lz30;
}"""))
    assert res.declarations[0] == Let("ztrace", NamedType("%bool"),
        [
            LocalVarDeclaration('zgsz30_lz30', NamedType('%bool')),
            Assignment('zgsz30_lz30', Var('false')),
            Assignment('ztrace', Var('zgsz30_lz30')),
        ])

def test_reftype():
    res = parser.parse(lexer.lex("""
val zzzzz7azzJzzKTLBEntryz3__deref = "reg_deref" : (&(%struct zTLBEntry)) ->  %struct zTLBEntry
"""))
    assert res.declarations[0].typ.argtype == TupleType([RefType(StructType('zTLBEntry'))])

def test_vectype():

    res = parser.parse(lexer.lex("""
register zGPR : %vec(%bv64)
"""))
    assert res.declarations[0].typ == VecType(NamedType("%bv64"))

def test_globalval():
    res = parser.parse(lexer.lex("""
val zeq_bool = "eq_bool" : (%bool, %bool) ->  %bool
val zneq_int : (%i, %i) ->  %bool

"""))
    assert res.declarations == [
        GlobalVal('zeq_bool', '"eq_bool"', FunctionType(TupleType([NamedType('%bool')] * 2), NamedType('%bool'))),
        GlobalVal('zneq_int', None, FunctionType(TupleType([NamedType('%i')] * 2), NamedType('%bool'))),
    ]

def test_register():
    res = parser.parse(lexer.lex("""
register zPC : %bv16
"""))
    assert res.declarations == [
        Register('zPC', NamedType('%bv16'))
    ]

def test_function():
    res = parser.parse(lexer.lex("""
fn zneq_int(zx, zy) {
  zgaz30_lz30 : %bool;
  zgaz30_lz30 = zeq_int(zx, zy);
  return = znot_bool(zgaz30_lz30);
  end;
}
"""))
    assert res.declarations[0] == Function(
        'zneq_int', ['zx', 'zy'],
        [
            LocalVarDeclaration('zgaz30_lz30', NamedType('%bool')),
            Operation('zgaz30_lz30', 'zeq_int', [Var('zx'), Var('zy')]),
            Operation('return', 'znot_bool', [Var('zgaz30_lz30')]),
            End(),
        ]
    )

def test_function2():
    res = parser.parse(lexer.lex("""
fn zsail_mask(zlen, zv) {
  zgaz32_lz30 : %bool;
  zgaz31_lz31 : %i;
  zgaz31_lz31 = zbitvector_length(zv);
  zgaz32_lz30 = zlteq_int(zlen, zgaz31_lz31);
  jump zgaz32_lz30 goto 7 ` "/home/cfbolz/.opam/default/share/sail/lib/vector_dec.sail 81:29 - 81:100";
  return = zsail_zzero_extend(zv, zlen);
  goto 8;
  return = ztruncate(zv, zlen);
  end;
}
"""))
    assert res.declarations[0] == Function(
        'zsail_mask', ['zlen', 'zv'],
        [
            LocalVarDeclaration('zgaz32_lz30', NamedType('%bool')),
            LocalVarDeclaration('zgaz31_lz31', NamedType('%i')),
            Operation('zgaz31_lz31', 'zbitvector_length', [Var('zv')]),
            Operation('zgaz32_lz30', 'zlteq_int', [Var('zlen'), Var('zgaz31_lz31')]),
            ConditionalJump(ExprCondition(Var('zgaz32_lz30')), 7, '"/home/cfbolz/.opam/default/share/sail/lib/vector_dec.sail 81:29 - 81:100"'),
            Operation('return', 'zsail_zzero_extend', [Var('zv'), Var('zlen')]),
            Goto(8),
            Operation('return', 'ztruncate', [Var('zv'), Var('zlen')]),
            End(),
        ]
    )

def test_function3():
    res = parser.parse(lexer.lex("""
fn zbits1_to_bool(zb) {
  zgsz310_lz30 : %bool;
  zb__0_lz33 : %bv1;
  zb__0_lz33 = zb;
  zgsz311_lz34 : %bool;
  zgsz311_lz34 = @eq(zb__0_lz33, 0b1);
  jump @not(zgsz311_lz34) goto 7 ` "f.sail 13:27 - 16:1";
  goto 8;
  goto 10;
  zgsz310_lz30 = true;
  goto 20;
  zb__1_lz31 : %bv1;
  zb__1_lz31 = zb;
  zgsz312_lz32 : %bool;
  zgsz312_lz32 = @eq(zb__1_lz31, 0b0);
  jump @not(zgsz312_lz32) goto 16 ` "f.sail 13:27 - 16:1";
  goto 17;
  goto 19;
  zgsz310_lz30 = false;
  goto 20;
  failure;
  return = zgsz310_lz30;
  end;
}
    """))
    assert res.declarations[0].body[2] == Assignment('zb__0_lz33', Var('zb'))
    assert res.declarations[0].body[5] == ConditionalJump(
            Comparison('@not', [Var('zgsz311_lz34')]), 7, '"f.sail 13:27 - 16:1"')

def test_function4():
    res = parser.parse(lexer.lex("""
fn zdecode(zmergez3var) {
  zgsz344_lz30 : %union zoption;
  zv__1_lz315 : %bv16;
  zv__1_lz315 = zmergez3var;
  zgaz317_lz316 : %bv1;
  zgaz317_lz316 = @slice::<1>(zv__1_lz315, 15);
  zgsz348_lz317 : %bool;
  zgsz348_lz317 = @eq(zgaz317_lz316, 0b0);
  jump @not(zgsz348_lz317) goto 9 ` "f.sail 97:16 - 98:39";
  goto 10;
  goto 24;
  zx_lz318 : %bv15;
  zx_lz318 = @slice::<15>(zv__1_lz315, 0);
  zgaz316_lz319 : %union zinstr;
  zgaz315_lz321 : %bv16;
  zgsz345_lz322 : %bv = zx_lz318;
  zgsz346_lz323 : %i = 16;
  zgsz347_lz324 : %bv;
  zgsz347_lz324 = zsail_zzero_extend(zgsz345_lz322, zgsz346_lz323);
  zgaz315_lz321 = zgsz347_lz324;
  zgaz316_lz319 = zAINST(zgaz315_lz321);
  zgsz398_lz320 : %union zinstr;
  zgsz398_lz320 = zgaz316_lz319;
  zgsz344_lz30 = zSomez3z5unionz0zzinstr(zgsz398_lz320);
  goto 61;
  zv__3_lz31 : %bv16;
  zv__3_lz31 = zmergez3var;
  zgaz323_lz32 : %bv3;
  zgaz323_lz32 = @slice::<3>(zv__3_lz31, 13);
  zgsz350_lz33 : %bool;
  zgsz350_lz33 = @eq(zgaz323_lz32, 0b111);
  jump @not(zgsz350_lz33) goto 32 ` "f.sail 97:16 - 98:39";
  goto 33;
  goto 60;
  zjump_lz34 : %bv3;
  zjump_lz34 = @slice::<3>(zv__3_lz31, 0);
  zdest_lz35 : %bv3;
  zdest_lz35 = @slice::<3>(zv__3_lz31, 3);
  zc_lz36 : %bv6;
  zc_lz36 = @slice::<6>(zv__3_lz31, 6);
  za_lz37 : %bv1;
  za_lz37 = @slice::<1>(zv__3_lz31, 12);
  zgaz322_lz38 : %union zinstr;
  zgaz321_lz310 : (%bv1, %enum zarithmetic_op, (%bool, %bool, %bool), %enum zjump);
  zgaz318_lz311 : %enum zarithmetic_op;
  zgaz318_lz311 = zdecode_compute_backwards(zc_lz36);
  zgaz319_lz312 : (%bool, %bool, %bool);
  zgaz319_lz312 = zdecode_destination(zdest_lz35);
  zgaz320_lz313 : %enum zjump;
  zgaz320_lz313 = zdecode_jump_backwards(zjump_lz34);
  zgsz349_lz314 : (%bv1, %enum zarithmetic_op, (%bool, %bool, %bool), %enum zjump);
  zgsz349_lz314.0 = za_lz37;
  zgsz349_lz314.1 = zgaz318_lz311;
  zgsz349_lz314.2 = zgaz319_lz312;
  zgsz349_lz314.3 = zgaz320_lz313;
  zgaz321_lz310 = zgsz349_lz314;
  zgaz322_lz38 = zCINST(zgaz321_lz310);
  zgsz399_lz39 : %union zinstr;
  zgsz399_lz39 = zgaz322_lz38;
  zgsz344_lz30 = zSomez3z5unionz0zzinstr(zgsz399_lz39);
  goto 61;
  zgsz344_lz30 = zNone(());
  return = zgsz344_lz30;
  end;
}"""))

def test_ztup_bug():
    res = parser.parse(lexer.lex("""
val zassign_dest : ((%bool, %bool, %bool), %bv16) ->  %unit

fn zassign_dest(zgsz371, zvalue) {
  za_lz30 : %bool;
  za_lz30 = zgsz371.ztup0;
  end;
}
    """))
    assert res.declarations[1].body[1] == Assignment(result='za_lz30', value=FieldAccess(Var(name='zgsz371'), 'ztup0'))


def test_dot_vs_is():
    res = parser.parse(lexer.lex("""
fn zexception_handler(zcur_priv, zctl, zpc) {
  ze_lz374 : %struct zsync_exception;
  ze_lz374 = zgsz34019_lz30.ztup1 as zCTL_TRAP;
  end;
}
"""))
    assert res.declarations[0].body[1].value == Cast(expr=FieldAccess(Var(name='zgsz34019_lz30'), 'ztup1'), variant='zCTL_TRAP')

def test_parse_full():
    with open(cir, "rb") as f:
        s = f.read()
    try:
        res = parser.parse(lexer.lex(s))
    except LexingError as e:
        print s[e.getsourcepos().idx:e.getsourcepos().idx+20]
        assert 0

