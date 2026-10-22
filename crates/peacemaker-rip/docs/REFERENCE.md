# `.rip` language reference

This reference targets RDNA gfx1151 and gfx1201, wave32. gfx1100 uses the gfx11 instruction spellings below. `.rip` is an evaluated kernel-authoring language, not rustc and not device Rust.

## 1. Mental model

The host parses the file with `syn`, evaluates `emit`, and records device instructions in `peacemaker_isa::Builder`. A script `for` loop runs while emitting: it unrolls instructions. A device loop must use the Workgroup driver's loop operations. A script `if arch.gfx12()` chooses instructions while emitting; a device conditional uses a driver condition.

The runtime Driver checks LDS ownership, publication, draining and workgroup-uniform barriers and control flow. The builder checks register ownership, instruction policy and memory hazards, and inserts dependency waits. Instruction validation checks the mnemonic and operands for the selected architecture and rejects incorrect declared register effects (R0603). Native assembly/lifting checks for registers read but neither entry-initialized nor written anywhere in the kernel (R0604), including an incorrect s2 workgroup-ID read on gfx1201. This check is path-free: it does not prove that a write precedes each read, covers each branch, or initializes both halves. Those stronger checks belong to M7 certification.

`check` executes the selected entry/variant for each selected architecture and reports these errors and warnings plus register/LDS/instruction counts. It does not prove the numerical algorithm, bounds for arbitrary runtime inputs, launch correctness or emulator coverage. Unexecuted script paths are not checked. Assembly/link certification is a separate `build --certify` step.

## 2. Files, entries and variants

The preferred entry signature is `fn emit(spec: Spec) -> Result<Emitted, String>`. The injected `Spec` has `arch: Arch`, `variant: String`, and `kind`. Declare a unit `enum Kind { Default, Other }` to make `spec.kind` the matching enum variant (case-insensitive matching against `spec.variant`); without a matching variant, `kind` is the variant string. Match `spec.kind` to choose emission routines. The alternate entry is `fn emit(arch: Arch, variant: String) -> Result<Emitted, String>`; it receives the same architecture and variant string separately. Finish with `builder.finish()`, not `Ok(())`.

`Arch::{Gfx1100,Gfx1151,Gfx1201}` are available. `arch.gfx12() -> bool`, `arch.name() -> String`, and `arch.buffer_offset_max() -> u32` inspect it; for example `let name = spec.arch.name();`.

`check` defaults to gfx1151 and gfx1201; `--arch all` includes gfx1100. Pass `--variant default` explicitly when there is no `Kind`. When omitted, the CLI uses the script's declared `Kind` variants. Variant and architecture selection happen at emission time, not at kernel launch.

## 3. The evaluated Rust subset

Supported executed syntax includes functions, const/static values, structs, enums (unit and tuple variants and discriminants), impl methods and trait defaults, lexical closures, tuples, arrays/Vec, Option/Result, `?`, references to mutable places, field/index assignment, `let else`, `if let`, `while let`, match guards/ranges/or-patterns/bindings, and labelled `for`/`while`/`loop` with break/continue. Generics, lifetimes and most type annotations are accepted but are not a rustc type system; integer annotations do matter. `use` and unused items are not resolved as Rust imports. Native APIs are a fixed vocabulary, not arbitrary crate access.

Unsupported executed syntax is a located error, never a silently ignored operation. Do not assume async, unsafe blocks, arbitrary standard-library APIs or user macro expansion work. Iterator adaptors are eager/materialized, not lazy streams; range materialization is capped at 2^22 elements. Keep emission-time collections small.

Integer types are `u8/u16/u32/u64/usize/i8/i16/i32/i64/isize`. Unsuffixed literals adopt the other operand's or declared parameter/binding's integer type. Mixed typed operands are errors; arithmetic overflow, zero division and invalid shifts are errors. Integer `as` casts truncate to the destination width; widening can use `u32::from(x)`.

```rip eval
fn main() -> u32 {
    let n: u8 = 200;
    assert_eq!(n + 50, 250u8);
    let low = 0x1ffu32 as u8;
    assert_eq!(low, 255u8);
    u32::from(low) + 1
}
```

Macros include `format!`, `vec!`, `matches!`, `assert!`, `assert_eq!`, `assert_ne!`, `panic!`, `unreachable!`, `todo!`, `unimplemented!`, and print/println/eprint/eprintln. Panic-like macros abort evaluation; they do not emit GPU traps. `format!` supports positional, named and captured arguments, escaped braces, Display/Debug (`?`, `#?`), radix `x/X/b/o`, exponent `e`, fill/alignment/sign/alternate/zero padding, fixed width and precision. Dynamic `$` width/precision arguments are unsupported.

```rip eval
fn main() -> String {
    let n = 255u32;
    let values: Vec<u32> = [1u32, 2, 3].into_iter().map(|x| x + 1).collect();
    assert_eq!(values[2], 4u32);
    format!("{0:#x} {n:08X} {values:?}", n)
}
```

## 4. Item keywords

These are token-level item rewrites, not extra runtime objects:

| Syntax | Becomes | Example |
|---|---|---|
| `region R;` or `region R, S;` | empty enum tag(s) | `region Scratch;` |
| `tile T { fields }` | `struct T { fields }` | `tile Pair { x: u8, y: u8 }` |
| `chain f(args) -> Ret { body }` | `fn f(args) -> Ret { body }` | `chain inc(x: u32) -> u32 { x + 1 }` |

`region` and `tile` are top-level items; `chain` also works inside impl/trait/module items. Attributes and visibility remain attached. A variable named `tile`, a `.chain()` method or a string containing these words is unaffected. A tile does not allocate registers; a region tag does not allocate LDS. Use RegPlan and `wg.lds` respectively.

```rip eval
region Scratch;
tile Pair { x: u32, y: u32 }
chain total(p: Pair) -> u32 { p.x + p.y }
fn main() -> u32 { total(Pair { x: 4u32, y: 9u32 }) }
```

## 5. KernelSpec and kernargs

`KernelSpec { ... }` has no omitted/default fields:

| Field | Type | Meaning |
|---|---|---|
| `kernel_id` | String | logical kernel identity |
| `variant` | String | emitted variant identity |
| `arch` | Arch | target architecture |
| `symbol` | String | assembly entry symbol |
| `kernargs` | KernargLayout | argument metadata/layout |
| `user_sgpr_count` | u8 | user SGPR count; use 2 for the kernarg pointer |
| `system_sgpr_workgroup_id_y` | bool | enable workgroup y ID |
| `workgroup_size` | u16 | metadata maximum flat workgroup size |
| `group_segment_fixed_size` | u32 | statically allocated LDS bytes |
| `wave32` | bool | must be true (`builder requires wave32`) |
| `cu_mode` | bool | CU mode; metadata WGP mode is its inverse |

`KernargLayout::new(size: u32) -> KernargLayout` sets total bytes, including padding; example `KernargLayout::new(16)`. `.pointer(name: str, offset: u32) -> KernargLayout` adds an eight-byte global pointer; example `.pointer("input", 0)`. `.hidden(name: str, offset: u32, size: u32, value_kind: str) -> KernargLayout` adds explicit-sized metadata; example `.hidden("count", 8, 4, "by_value")`. `.validate() -> Result<(), String>` checks layout bounds/overlap; example `layout.validate()?`. These are metadata, not argument loads; offsets and total size must match the host's packed argument bytes. Metadata kernarg alignment is eight bytes.

`smem(b, dst: u8, len: u8, base: u8, off: u32) -> Result<(), String>` loads `len` dwords from the 64-bit pointer in `s[base:base+1]` plus byte offset `off`. Allowed widths are 1/2/4/8, emitting `s_load_b32/b64/b128/b256`; other widths return `SMEM width`. The **entire destination range must fit one live plan allocation**. Four separately planned SGPRs do not own a four-dword load.

```rip body
smem(wg.isa(), 4, 4, 0, 0)?;
// s[4:7] is the single "args" allocation in the template.
op(wg.isa(), "v_mov_b32_e32 v1, s4", &[v(1)], &[s(4)])?;
```

## 6. Register plans

`RegPlan::new(vgpr_budget: u16, sgpr_budget: u16) -> Result<RegPlan, String>` accepts budgets up to 256 VGPRs and 104 SGPRs; example `let mut p = RegPlan::new(256, 104)?;`. A budget is a limit, not an allocation.

`p.s::<N>(name: str, base: u8, live: Live) -> Result<S<N>, String>` and `p.v::<N>(...) -> Result<V<N>, String>` allocate a named contiguous range; example `p.s::<4>("args", 4, Live::Whole)?;`. Only N = 1, 2, 4, 8 are supported (`register width must be 1, 2, 4 or 8`). Multi-dword scalar allocations require base alignment to their width. Register names in instruction strings are still physical registers, not these symbolic allocation names.

`Live::Whole` covers the whole kernel. `Live::Between(start: str, end: str)` covers a start-inclusive, end-exclusive label interval; example `Live::Between(".Lload", ".Ldone")`. Place those boundaries using `wg.label`. Do not reuse physical storage across overlapping lifetimes. `p.next_free_vgpr() -> u16` and `p.next_free_sgpr() -> u16` return the high-water marks; example `let top = p.next_free_vgpr();`. `p.scratch_pool(base: u8, len: u8) -> Result<(), String>` configures temporary VGPR storage; example `p.scratch_pool(128, 8)?;`.

Every declared access must have exactly one live owner containing the **whole access range**, not merely one owner per component. The exact diagnostic is, for example, `s[12:15] at entry has no unique live owner` (R0401). Fix a four-dword load by replacing four single-register allocations with `p.s::<4>("args", 12, Live::Whole)?;`, or split the actual load into owned widths. Ownership is checked for defs and uses, including incoming ABI registers you read.

Manual VGPR granule padding is **not required by RegPlan or the emitter**: the emitted count is the plan's high-water mark; the native wave32 descriptor encoder rounds its allocation encoding in units of eight. Some production scripts add an unused high-register owner to round metadata counts to eight; this is a convention, not a language requirement. Do not confuse descriptor encoding units with hardware occupancy/allocation granules.

## 7. Instruction helpers

`b` below is either a Builder or `wg.isa()`; register indices are u8, byte offsets u32. Helpers returning Result require `?`. Defs are registers written; uses are registers read, including read-modify-write destinations. These declarations drive automatic waits, register ownership and source-lock/hazard checks. Omitting an operand can drop a required wait; `check` rejects it with R0603. `MemoryClass::{VmemLoad,VmemStore,DsLoad,DsStore,SmemLoad,Export}` identifies the asynchronous operation family.

| Signature / return | Emission / example |
|---|---|
| `v(n), s(n) -> RegRef` | one VGPR/SGPR; `v(3)`, `s(4)` |
| `vr(base,len), sr(base,len) -> RegRef` | contiguous range (second argument is length, not last index); `sr(8,4)` is s[8:11] |
| `lit(x: u32) -> String` | canonical decimal for 0..64 and bit patterns -16..-1, hex otherwise; `lit(128)` is `"0x80"` |
| `op(b,text,defs:&[RegRef],uses:&[RegRef]) -> Result<(),String>` | given instruction text; `op(wg.isa(),"v_mov_b32_e32 v1, 0",&[v(1)],&[])?` |
| `mem(b,text,defs,uses,class) -> Result<(),String>` | given text with memory classification; `mem(wg.isa(),text,&[v(1)],&[v(0),sr(8,4)],MemoryClass::VmemLoad)?` |
| `sop(b,text,defs:&[u8],uses:&[u8]) -> Result<(),String>` | scalar single-register shorthand; `sop(wg.isa(),"s_mov_b32 s12, 0",&[12],&[])?` |
| `smem(b,dst,len,base,off) -> Result<(),String>` | scalar load, described above; `smem(wg.isa(),4,4,0,0)?` |
| `srd_tail(b,srd,records:Option<u8>) -> Result<(),String>` | writes descriptor words 2/3; records from SGPR or -1 for None; `srd_tail(wg.isa(),8,None)?` |
| `add64(b,dst,src,x) -> Result<(),String>` | SGPR-pair dst = src + scalar s[x], carry propagation; `add64(wg.isa(),12,4,6)?` |
| `add64_imm(b,reg,imm:u32) -> Result<(),String>` | increments an SGPR pair, using `lit`; `add64_imm(wg.isa(),12,128)?` |
| `bload(b,width,dst,voff,srd,offset) -> Result<(),String>` | buffer load width 1/2/4 dwords at v[voff]+offset; `bload(wg.isa(),1,1,0,8,0)?` |
| `bstore_b128(b,data,voff,srd,offset) -> Result<(),String>` | four-dword buffer store; `bstore_b128(wg.isa(),4,0,8,16)?` |
| `s_add_u32(arch), s_addc_u32(arch), s_add_i32(arch) -> str` | architecture-specific opcode strings, table below; `let add = s_add_u32(arch);` |
| `smem_off(arch,off:u32) -> String` | hex offset; zero is `null` on gfx11, `0x0` on gfx12; `smem_off(arch,0)` |
| `Instruction::new(text,defs,uses) -> Instruction` | instruction value for driver APIs; `Instruction::new("s_cmp_eq_u32 s12, 0", &[], &[s(12)])` |
| `insn.memory(class) -> Instruction` | classify instruction; `insn.memory(MemoryClass::DsStore)` |
| `workgroup_id(b,axis:u8,dst:u8) -> Result<(),String>` | read x/y (axis 0/1) into s[dst] using the architecture's ABI; `workgroup_id(wg.isa(),0,12)?;` |

`SRD_WORD3: u32 = 0x31004000` selects raw untyped 32-bit buffer accesses: offsets at/past num_records read zero without touching memory. `srd_tail` uses this constant. Buffer helpers use `offen`, zero soffset (`0` gfx11, `null` gfx12), and byte immediate offsets, limited to 4095 on gfx11 and 8388607 on gfx12. They do not construct descriptor base words 0/1. Buffer and global operations are different families of addressing instructions; do not replace one spelling with the other.

`gather_offset(b,dst:u8,index:u8,div:Option<u8>,row_bytes:u32,live:u8,t:GatherTemps) -> Result<(),String>` computes signed indexed-row byte offsets and live-lane mask. `GatherTemps { v:[u8;4], mask:u8 }` supplies four dead VGPRs and scalar mask scratch. Example: `gather_offset(wg.isa(),1,0,None,16,12,GatherTemps { v:[8,9,10,11], mask:13 })?;`. Live indices must be nonnegative and below `GATHER_MAX_INDEX = 1<<22`; negative padding indices become `GATHER_OOB = 0x7fffff00`. A supplied divisor is an SGPR index (positive uniform divisor); None means one. `row_bytes` must be below 2^24. Set the descriptor's record count to the actual byte extent; the helper emits reciprocal division plus integer corrections, offset multiplication and mask selection, not a load.

`Chain::new(dst:u8,width:u8,kind:str) -> Chain` describes a WMMA accumulator; width must be 8 and kind `"f16"` or `"bf16"`. Example `let c = Chain::new(0,8,"f16");`. `c.step(b,a:u8,aw:u8,b:u8,bw:u8) -> Result<(),String>` emits the kind's `v_wmma_f32_16x16x16_*` using and redefining its accumulator; example `c.step(wg.isa(),8,8,16,8)?;` on gfx11. Operand widths must match the architecture (eight on gfx11, four on gfx12); initialize the accumulator before its first step. The item keyword `chain` merely defines a function and is unrelated to this native Chain value.

`Wmma::swmmac(arch, kind: Swmmac, dst, a, b, index: RegRef, mods: SwmmacMods) -> Result<Instruction,String>` builds a gfx12 sparse (2:4) WMMA; push it with `wg.isa().push(..)?`. `kind` is `Swmmac::<Kind>`; `dst`, `a` and `b` are `vr(..)` ranges of exactly the kind's width (table below); `index` is one VGPR (`v(n)`); `mods` is the struct literal `SwmmacMods { index_key, neg_lo, neg_hi, clamp }` with all four fields written (`u8, u8, u8, bool`). SWMMAC has no C operand: `dst` is read in place as the accumulator, so the instruction declares defs `[dst]` and uses `[a, b, index, dst]`, and the accumulator must be initialized first. Example: `wg.isa().push(Wmma::swmmac(arch, Swmmac::F32F16, vr(24,8), vr(8,4), vr(12,8), v(20), SwmmacMods { index_key: 0, neg_lo: 0, neg_hi: 0, clamp: false })?)?;` emits `v_swmmac_f32_16x16x32_f16 v[24:31], v[8:11], v[12:19], v20`. Refusals carry the Rust error text: `SWMMAC requires gfx12` (gfx1100/gfx1151), `<mnemonic>: A requires 4 VGPRs, found v[8:10]` (width), and the modifier errors below. A hand-written `op` with the same text must declare `dst` in its uses as well. Worked example: `docs/cookbook/10-swmmac.md`.

| `Swmmac::` kind | mnemonic | D | A | B | modifiers |
|---|---|---|---|---|---|
| `F32F16` | v_swmmac_f32_16x16x32_f16 | 8 | 4 | 8 | float |
| `F32Bf16` | v_swmmac_f32_16x16x32_bf16 | 8 | 4 | 8 | float |
| `F16F16` | v_swmmac_f16_16x16x32_f16 | 4 | 4 | 8 | float |
| `Bf16Bf16` | v_swmmac_bf16_16x16x32_bf16 | 4 | 4 | 8 | float |
| `I32Iu8` | v_swmmac_i32_16x16x32_iu8 | 8 | 2 | 4 | int |
| `I32Iu4` | v_swmmac_i32_16x16x32_iu4 | 8 | 1 | 2 | int |
| `I32Iu4X64` | v_swmmac_i32_16x16x64_iu4 | 8 | 2 | 4 | int, no `index_key` |
| `F32Fp8Fp8` | v_swmmac_f32_16x16x32_fp8_fp8 | 8 | 2 | 4 | fp8 |
| `F32Fp8Bf8` | v_swmmac_f32_16x16x32_fp8_bf8 | 8 | 2 | 4 | fp8 |
| `F32Bf8Fp8` | v_swmmac_f32_16x16x32_bf8_fp8 | 8 | 2 | 4 | fp8 |
| `F32Bf8Bf8` | v_swmmac_f32_16x16x32_bf8_bf8 | 8 | 2 | 4 | fp8 |

Counts are VGPRs (dwords); the index is always one VGPR. Legal modifier sets (as accepted by `llvm-mc -mcpu=gfx1201`; anything else is refused as `<mnemonic> does not accept <modifier>`, `<mnemonic> has no index_key` or `<mnemonic>: neg_lo/neg_hi bit 2 (C) is not encodable`):

| class | `index_key` | `neg_lo` | `neg_hi` | `clamp` |
|---|---|---|---|---|
| float (`F32F16` `F32Bf16` `F16F16` `Bf16Bf16`) | 0 or 1 | bits 0-1 (A, B) | bits 0-1 | no |
| int (`I32Iu8` `I32Iu4`) | 0 or 1 | bits 0-1 | no | yes |
| `I32Iu4X64` | no (must be 0) | bits 0-1 | no | yes |
| fp8 (`F32Fp8Fp8` `F32Fp8Bf8` `F32Bf8Fp8` `F32Bf8Bf8`) | 0 or 1 | no | no | no |

`neg_lo`/`neg_hi` bit 0 negates A and bit 1 negates B; bit 2 (C) never encodes. Text order is `index_key:1 neg_lo:[a,b,0] neg_hi:[a,b,0] clamp`; `index_key:1` is OP_SEL bit 0 (`index_key:0` is the default and is not printed). In instruction text `index_key:N` is accepted only on `v_swmmac_*` rows other than `v_swmmac_i32_16x16x64_iu4`.

## 8. Workgroup driver

### Scope and complete skeleton

`let mut wg = Workgroup::new(&mut builder, arch)?;` must be a let initializer. Its scope ends at the first later statement mentioning the **builder variable's name**, even `let builder = wg.isa();`. Subsequent access fails with `wg.isa() outside the Workgroup scope` (R0305). Avoid borrowing an ISA alias in the owning block; call `wg.isa()` directly. Recommended: construct the Workgroup in `emit`, then call a separate author function. There, `let b = wg.isa();` does not mention the original `builder` identifier.

The following complete script is also the template for every `rip body` block on this page: splice statements at the comment. It supplies s0..s15 and v0..v15, but do not read uninitialized scratch values in a real kernel.

```rip
fn emit(spec: Spec) -> Result<Emitted, String> {
    let arch = spec.arch;
    let kspec = KernelSpec { kernel_id: "doc".into(), variant: "default".into(), arch, symbol: "doc_kernel".into(),
        kernargs: KernargLayout::new(16).pointer("a", 0).pointer("b", 8), user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false, workgroup_size: 64, group_segment_fixed_size: 4096, wave32: true, cu_mode: false };
    let mut p = RegPlan::new(256, 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    if !arch.gfx12() { p.s::<1>("wg_x", 2, Live::Whole)?; }
    p.s::<4>("args", 4, Live::Whole)?; p.s::<4>("srd", 8, Live::Whole)?; p.s::<4>("t", 12, Live::Whole)?;
    p.v::<8>("v", 0, Live::Whole)?; p.v::<8>("w", 8, Live::Whole)?;
    let mut builder = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut builder, arch)?;
    author(&mut wg, arch)?;
    builder.finish()
}
region Doc;
fn author<T: Target>(wg: &mut Wg<T>, arch: Arch) -> Result<(), String> {
    let end = wg.exit(".Ldoc_end")?;
    /*BODY*/
    wg.end(end)
}
```

### LDS ownership and tokens

These signatures describe script values, which sometimes differ from the Rust runtime API. `Result<X>` below abbreviates `Result<X,String>`. A Place is a region, ring or its Pending store token. Region states are Free → Writing → Published → Free. A wait drains operations; a barrier publishes or retires ownership. A barrier is not a memory-counter wait.

| Script signature / return | Meaning and one-line example |
|---|---|
| `wg.isa() -> Isa` | checked instruction handle; `op(wg.isa(),text,defs,uses)?` |
| `Workgroup::new(builder,arch) -> Result<Workgroup>` | open scoped driver, must initialize let; `let mut wg = Workgroup::new(&mut builder,arch)?;` |
| `wg.position() -> usize` | instruction position; `let at = wg.position();` |
| `wg.lds::<R>(name:str,base:u32,len:u32) -> Result<Region>` | allocate LDS bytes with tag R; `let r = wg.lds::<Doc>("scratch",0,256)?;` |
| `wg.begin_write(p:Place) -> (Place,Pending)` | mark Writing without an instruction; **no `?`**; `let (r,p) = wg.begin_write(r);` |
| `wg.ds_store(p,insn) -> Result<(Place,Pending)>` | store to Free/Writing; `let (r,p) = wg.ds_store(r,insn)?;` |
| `wg.ds_store2(x,y,insn) -> Result<((Place,Pending),(Place,Pending))>` | one instruction stores to two distinct places; `let ((x,px),(y,py)) = wg.ds_store2(x,y,insn)?;` |
| `wg.ds_load(r,insn) -> Result<()>` | load Published region; `wg.ds_load(r,insn)?;` |
| `wg.ds_load_cur(g,insn) -> Result<()>` | load ring current Published buffer; `wg.ds_load_cur(g,insn)?;` |
| `wg.wait(p) -> Result<Drained>` | drain a Writing place's stores (not a Published place's loads); `let d = wg.wait(p)?;` |
| `wg.wait_all((p,q,...)) -> Result<(Drained,Drained,...)>` | same-order drained tuple; `let (dp,dq) = wg.wait_all((p,q))?;` |
| `ready(r[,drained]) -> Transition` | Writing→Published request; `ready(r,d)` |
| `retire(r[,drained]) -> Transition` | Published→Free request; no store wait needed; `retire(r)` |
| `wg.barrier((transition,...)) -> Result<(handle,...)>` | apply transitions in order, same-arity tuple; `let (r,) = wg.barrier((ready(r,d),))?;` |
| `wg.signal((transition,...)) -> Result<Arrived>` | gfx1201 split arrival, holds affected handles; `let a = wg.signal((ready(r,d),))?;` |
| `wg.wait_arrived(a:Arrived) -> Result<()>` | complete split barrier and apply transitions; `wg.wait_arrived(a)?;` |
| `wg.join((r,s,...)) -> Result<Region>` | consume ≥2 Free regions, up to four slots; `let rs = wg.join((r,s))?;` |
| `join((r,s,...)) -> Result<Region>` | free-function spelling; `let rs = join((r,s))?;` |
| `wg.split(r,first:usize) -> Result<(Region,Region)>` | split Free joined region at slot count, not byte offset; `let (r,s) = wg.split(rs,1)?;` |
| `wg.ring(name,x,y) -> Result<Ring>` | consume distinct Free regions; `let g = wg.ring("pingpong",x,y)?;` |
| `Ring::new(x,y) -> Result<Ring>` | unnamed ring construction; `let g = Ring::new(x,y)?;` |
| `g.into_steady() -> Result<Ring>` | mark current buffer vacant for steady rotation; `let g = g.into_steady()?;` |
| `wg.ring_steady(g) -> Result<Ring>` | same steady-ring operation; `let g = wg.ring_steady(g)?;` |
| `g.cur_index(), g.next_index() -> usize` | current/next backing indices; `let i = g.cur_index();` |
| `g.phases() -> (Phase,Phase)` | backing phases; `let phases = g.phases();` |
| `r.phase() -> Phase` | Free/Writing/Published; `let phase = r.phase();` |
| `prime(g[,d]), rotate(g[,d]), retire_cur(g[,d]) -> Transition` | ring barrier transitions; `let (g,) = wg.barrier((rotate(g,d),))?;` |
| `wg.relayout() -> Result<()>` | require all Free; invalidate old handles; `wg.relayout()?;` |

Optional Drained tokens document the operation you waited for, but the driver also checks its tracked youngest store: old evidence cannot cover a newer store. Use the tokens returned by the latest operations. `signal/wait_arrived` are gfx1201 only; gfx11 has one `s_barrier`, so use `barrier`.

```rip body
let r = wg.lds::<Doc>("scratch", 0, 256)?;
op(wg.isa(), "v_lshlrev_b32_e32 v1, 2, v0", &[v(1)], &[v(0)])?;
let store = Instruction::new("ds_store_b32 v1, v0", &[], &[v(1),v(0)]).memory(MemoryClass::DsStore);
let (r, pending) = wg.ds_store(r, store)?;
let drained = wg.wait(pending)?;
let (r,) = wg.barrier((ready(r, drained),))?;
let load = Instruction::new("ds_load_b32 v2, v1", &[v(2)], &[v(1)]).memory(MemoryClass::DsLoad);
wg.ds_load(r, load)?;
op(wg.isa(), "v_xor_b32_e32 v3, 1, v2", &[v(3)], &[v(2)])?;
let (r,) = wg.barrier((retire(r),))?;
```

### Conditions, closures, loops and exits

Conditions record SCC provenance. Consume a fresh condition before an SCC-writing instruction intervenes. `scmp_wg_uniform` is the author's assertion that operands are workgroup-uniform (kernargs, workgroup IDs, constants, and derived counters), not a proof of that assertion.

All closure methods below return Results. Closures return `Ok(...)` or propagate with `?`. `R` denotes the closure's script result value. A closure emits code; it is not called once per runtime lane. Loop closures can be replayed to reach a builder fixed point: keep emission deterministic.

| Script signature / return | Semantics and one-line example |
|---|---|
| `wg.scmp(insn) -> Result<Cond>` | wave-uniform scalar compare; `let c = wg.scmp(Instruction::new("s_cmp_eq_u32 s12, 0",&[],&[s(12)]))?;` |
| `wg.scmp_wg_uniform(insn) -> Result<WgCond>` | author claims WG uniformity; `let c = wg.scmp_wg_uniform(insn)?;` |
| `wg.skip_if(c,label,\|w\| -> Result<()> {...}) -> Result<()>` | skip body when true; `wg.skip_if(c,".Lskip",\|w\| Ok(()))?;` |
| `wg.skip_unless(c,label,\|w\| -> Result<()> {...}) -> Result<()>` | skip body when false; `wg.skip_unless(c,".Lskip",\|w\| Ok(()))?;` |
| `wg.wg_skip_if(c:WgCond,label,\|w\| -> Result<()> {...}) -> Result<()>` | WG-uniform skip, barriers allowed; `wg.wg_skip_if(c,".Lskip",\|w\| Ok(()))?;` |
| `skip_if/skip_unless/wg_skip_if(c,label,state,\|w,st\| -> Result<S> {...}) -> Result<S>` | explicit script state input/output; `let st = wg.skip_if(c,".Lskip",st,\|w,st\| Ok(st))?;` |
| `wg.exec_if(c,\|w\| -> Result<R> {...}) -> Result<R>` | EXEC all/none from SCC, then restored to all; wave scope; `wg.exec_if(c,\|w\| Ok(()))?;` |
| `wg.if_else(c,else_label,join_label,then,els) -> Result<()>` | **true selects els**, false selects then; both receive w, return Result<()>; `wg.if_else(c,".Lelse",".Ljoin",\|w\| Ok(()),\|w\| Ok(()))?; wg.label(".Ljoin")?;` |
| `wg.forward(\|w,fw\| -> Result<R> {...}) -> Result<R>` | forward-only wave branches; `wg.forward(\|w,fw\| { fw.goto(".Lf")?; fw.place(".Lf")?; Ok(()) })?;` |
| `fw.branch_if(c,label), fw.branch_unless(c,label), fw.goto(label), fw.place(label) -> Result<()>` | every target placed once after branch; `fw.branch_if(c,".Lf")?;` |
| `wg.loop_until(head,exit,\|w,brk\| -> Result<()> {...}) -> Result<()>` | wave-scope device loop; `wg.loop_until(".Lhead",".Lbreak",\|w,brk\| { let c = w.scmp(insn)?; w.break_if(c,brk) })?;` |
| `wg.break_if(c,brk) -> Result<()>` | conditional exit from enclosing loop_until; `w.break_if(c,brk)?;` |
| `wg.loop_carried(head,\|w\| -> Result<WgCond> {...}) -> Result<()>` | WG loop, true back edge; `wg.loop_carried(".Lhead",\|w\| w.scmp_wg_uniform(insn))?;` |
| `wg.loop_carried(head,state,\|w,st\| -> Result<(S,WgCond)> {...}) -> Result<S>` | carry script state and WG condition; `let st = wg.loop_carried(".Lhead",st,\|w,st\| Ok((st,w.scmp_wg_uniform(insn)?)))?;` |
| `wg.exit(label) -> Result<Exit>` | reserve final kernel exit; `let end = wg.exit(".Lend")?;` |
| `wg.exit_if(c:WgCond,end) -> Result<()>` | whole WG exits when true; `wg.exit_if(c,&end)?;` |
| `wg.wave_exit_unless(c:Cond,end) -> Result<()>` | wave exits when false; **no later WG barrier**; `wg.wave_exit_unless(c,&end)?;` |
| `wg.wg_exit_unless(c:WgCond,target,end,\|w\| -> Result<()> {...}) -> Result<()>` | false runs body and exits; true continues at target; `wg.wg_exit_unless(c,".Lcontinue",&end,\|w\| Ok(()))?;` |
| `wg.end(end) -> Result<()>` | place exit and emit s_endpgm; `wg.end(end)?;` |
| `wg.end_with(end,\|w\| -> Result<()> {...}) -> Result<()>` | exit tail in wave scope; nonempty tail forbidden if any branch targets exit; `wg.end_with(end,\|w\| Ok(()))?;` |
| `wg.handoff(readers:Cond,label,end,r,write,read) -> Result<R>` | final wave-role phase: write receives `(w,r)` and returns Result<()>; read receives `(w,r,exit)` and returns Result<R>; `wg.handoff(c,".Lread",end,r,\|w,r\| { let _ = w.begin_write(r); Ok(()) },\|w,r,exit\| Ok(()))?;` |
| `wg.label(name:str) -> Result<()>` | straight-line lifetime boundary; `wg.label(".Lphase")?;` |

Wave-scope bodies (`skip_if`, `exec_if`, `forward`, `if_else`, `loop_until`) cannot contain workgroup barriers or change the LDS ownership shape across joining paths. Workgroup-uniform skips must likewise restore the skipped path's ownership shape. Loops must have matching entry/back-edge ownership, including drain evidence; `loop_until` needs at least one `break_if`. `handoff` starts with a Free region, writers must leave it Writing, and the driver performs wait/publication, joins readers at the barrier and ends the kernel. Do not call `end` again after handoff.

```rip body
sop(wg.isa(), "s_mov_b32 s12, 0", &[12], &[])?;
let c = wg.scmp(Instruction::new("s_cmp_eq_u32 s12, 0", &[], &[s(12)]))?;
wg.skip_if(c, ".Lskip_doc", |w| {
    op(w.isa(), "v_mov_b32_e32 v1, 7", &[v(1)], &[])?;
    Ok(())
})?;
```

## 9. RDNA launch ABI

| Entry value | gfx1100 / gfx1151 | gfx1201 |
|---|---|---|
| kernarg pointer | s[0:1], with user_sgpr_count=2 | same |
| workgroup x | first system SGPR after user SGPRs: s2 with count=2; plan must own it | ttmp9, not in RegPlan |
| workgroup y (when enabled) | next SGPR: s3 with count=2; plan must own it | low 16 bits of ttmp7 |
| workgroup z ABI packing | next enabled system SGPR in ABI order | high 16 bits of ttmp7 |
| workitem x / flat thread ID | v0 within the workgroup | same |
| wave/EXEC | wave32; incoming EXEC identifies active workitems | same |

The current descriptor always enables x, optionally y, and disables z: z is **not available** through this KernelSpec on either architecture. It enables only workitem x (not separate y/z VGPRs). This reference assumes one-dimensional workgroups, where incoming v0 is the flat workgroup-local workitem index; for multidimensional launches v0 contains x only. It is not a global thread index or necessarily a lane index; for a 1D launch derive the lane with `v0 & 31`. Own every SGPR/VGPR range you access; ttmp registers are special registers outside the plan. A scalar group ID does not vary per wave or lane. Entry EXEC identifies active workitems, not necessarily every lane of a final partial wave.

Prefer `workgroup_id(wg.isa(), 0, 12)?;` over hand-written s2/ttmp9 selection. It defines s12; on gfx11 its incoming system SGPR source still needs a plan owner. gfx12 axis 0 emits `s_mov_b32` from ttmp9, axis 1 emits `s_and_b32` from ttmp7 with 0xffff. gfx11 uses `s_mov_b32` from the system SGPR after the user SGPRs (plus one for y). Axis 1 requires `system_sgpr_workgroup_id_y: true`; axis 2 is refused on both architectures.

```rip body
workgroup_id(wg.isa(), 0, 12)?;
op(wg.isa(), "v_mov_b32_e32 v1, s12", &[v(1)], &[s(12)])?;
```

## 10. Architecture-specific instruction vocabulary

The tables are the supported PeaceMaker vocabulary, not a complete silicon ISA catalogue. Use exact suffixes: `v_add_nc_u32_e32` exists; bare `v_add_nc_u32` is not its alias.

| Operation | gfx1100/gfx1151 | gfx1201 |
|---|---|---|
| unsigned add / carry-in | s_add_u32 / s_addc_u32 | s_add_co_u32 / s_add_co_ci_u32 |
| signed add | s_add_i32 | s_add_co_i32 |
| subtract / borrow-in | s_sub_u32 / s_subb_u32 | s_sub_co_u32 / s_sub_co_ci_u32 |
| signed subtract | s_sub_i32 | s_sub_co_i32 |
| float min/max | v_min_f32_e32 / v_max_f32_e32 | v_min_num_f32_e32 / v_max_num_f32_e32 |
| float DPP max | not in current tables | not in current tables |
| zero buffer soffset | 0 | null |
| zero SMEM offset | null | 0x0 |
| scalar dependency wait | s_waitcnt_depctr | s_wait_alu |
| barrier | s_barrier | s_barrier_signal -1; s_barrier_wait 0xffff |
| immediate buffer byte offset max | 4095 | 8388607 |
| v_permlane16_b32 | in gfx11 tables | absent |
| v_permlanex16_b32 | available | available |
| DPP row_xmask / row_shl | v_mov_b32_dpp | v_mov_b32_dpp |
| DPP quad_perm:[...] syntax | rejected by current modifier parser | rejected by current modifier parser |
| f16 WMMA 16x16x16 operands | dst/acc 8 VGPRs, A/B 8 each | dst/acc 8 VGPRs, A/B 4 each |

Inline integer constants are -16..64. A larger integer is a literal, spelled canonically with `lit`, not a decimal inline constant. There is one literal word per instruction (reusing the same literal is different from requesting two distinct literal words). VOP2 `_e32` has VGPR-only src1: use `_e64` when that source must be scalar or constant, or commute operands if valid. Not every opcode has both encodings in the tables. DPP controls are instruction modifiers, not standalone cross-lane instructions. `row_shr` is not a supported substitute for `row_shl`.

The following checks prove representative spellings and intentional rejections. F16 WMMA and SWMMAC widths are in dwords, not element counts.

```mnemonics
gfx1151: s_add_u32 s12, s12, s13
gfx1201: s_add_co_u32 s12, s12, s13
gfx1151: s_addc_u32 s12, s12, 0
gfx1201: s_add_co_ci_u32 s12, s12, 0
gfx1151: s_add_i32 s12, s12, s13
gfx1201: s_add_co_i32 s12, s12, s13
gfx1151: s_sub_u32 s12, s12, s13
gfx1201: s_sub_co_u32 s12, s12, s13
gfx1151: s_subb_u32 s12, s12, 0
gfx1201: s_sub_co_ci_u32 s12, s12, 0
gfx1151: s_sub_i32 s12, s12, s13
gfx1201: s_sub_co_i32 s12, s12, s13
gfx1151: v_min_f32_e32 v1, v2, v3
gfx1151: v_max_f32_e32 v1, v2, v3
gfx1201: v_min_num_f32_e32 v1, v2, v3
gfx1201: v_max_num_f32_e32 v1, v2, v3
!gfx1151: v_max_f32_dpp v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
!gfx1201: v_max_num_f32_dpp v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
gfx1151: buffer_load_b32 v1, v0, s[8:11], 0 offen offset:4095
gfx1201: buffer_load_b32 v1, v0, s[8:11], null offen offset:8388607
gfx1151: s_load_b32 s4, s[0:1], null
gfx1201: s_load_b32 s4, s[0:1], 0x0
gfx1151: s_waitcnt_depctr 0
gfx1201: s_wait_alu 0
gfx1151: s_barrier
gfx1201: s_barrier_signal -1
gfx1201: s_barrier_wait 0xffff
gfx1151: v_permlane16_b32 v1, v2, s12, s13
!gfx1201: v_permlane16_b32 v1, v2, s12, s13
gfx1151: v_permlanex16_b32 v1, v2, s12, 0xfedcba98
gfx1201: v_permlanex16_b32 v1, v2, s12, 0xfedcba98
gfx1151: v_mov_b32_dpp v1, v2 row_xmask:1 row_mask:0xf bank_mask:0xf
gfx1201: v_mov_b32_dpp v1, v2 row_xmask:1 row_mask:0xf bank_mask:0xf
!gfx1151: v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
!gfx1201: v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
gfx1151: v_mov_b32_dpp v1, v2 row_shl:1 row_mask:0xf bank_mask:0xf
gfx1201: v_mov_b32_dpp v1, v2 row_shl:1 row_mask:0xf bank_mask:0xf
gfx1151: v_add_nc_u32_e32 v1, 0x80, v2
gfx1201: v_add_nc_u32_e32 v1, 0x80, v2
!gfx1151: v_add_nc_u32_e32 v1, v2, 0x80
!gfx1201: v_add_nc_u32_e32 v1, v2, 0x80
gfx1151: v_add_nc_u32_e64 v1, v2, 0x80
gfx1201: v_add_nc_u32_e64 v1, v2, 0x80
!gfx1151: v_add_nc_u32_e64 v1, 0x80, 0x81
!gfx1201: v_add_nc_u32_e64 v1, 0x80, 0x81
!gfx1151: v_add_nc_u32 v1, v2, v3
!gfx1201: v_add_nc_u32 v1, v2, v3
gfx1151: v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]
gfx1201: v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[16:19], v[0:7]
gfx1201: v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20
gfx1201: v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20 index_key:1 neg_lo:[1,1,0] neg_hi:[1,1,0]
gfx1201: v_swmmac_i32_16x16x32_iu8 v[0:7], v[8:9], v[10:13], v14 index_key:1 neg_lo:[1,1,0] clamp
gfx1201: v_swmmac_f32_16x16x32_fp8_fp8 v[0:7], v[8:9], v[10:13], v14 index_key:1
!gfx1201: v_swmmac_i32_16x16x64_iu4 v[0:7], v[8:9], v[10:13], v14 index_key:1
!gfx1151: v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20
gfx1151: s_waitcnt vmcnt(0)
gfx1151: s_waitcnt lgkmcnt(0)
gfx1151: s_waitcnt_vscnt null, 0
gfx1151: s_waitcnt expcnt(0)
gfx1201: s_wait_loadcnt 0
gfx1201: s_wait_storecnt 0
gfx1201: s_wait_dscnt 0
gfx1201: s_wait_kmcnt 0
```

## 11. Waits and counters

Declared defs/uses let the builder insert waits for SMEM→use, buffer-load→use, DS-load→use and source reuse hazards. For example the `smem` body example above needs no handwritten wait before reading s4. Do not add blanket waits after every load. Conversely, omitting uses is not a performance optimization: it suppresses dependency knowledge.

`b.wait(counter: Counter, n: u8) -> Result<(),String>` emits an explicit ordering wait; example `wg.isa().wait(if arch.gfx12() { Counter::Store } else { Counter::Vs },0)?;`. The count is the maximum outstanding count allowed after the wait, not a cycle delay. Use driver waits for LDS ownership/publication, not raw waits alone.

| Counter | gfx1100/gfx1151 | gfx1201 |
|---|---|---|
| vector loads | Vm → s_waitcnt vmcnt(n) | Load → s_wait_loadcnt n |
| vector stores | Vs → s_waitcnt_vscnt null,n | Store → s_wait_storecnt n |
| LDS | Lgkm → s_waitcnt lgkmcnt(n) | Ds → s_wait_dscnt n |
| scalar memory | Lgkm (shared with LDS) | Km → s_wait_kmcnt n |
| export | Exp → s_waitcnt expcnt(n) | ledger accepts Exp but formats legacy s_waitcnt; avoid for gfx12 |

Passing a counter from the other architecture returns exactly `counter/architecture mismatch` (R0404). The gfx12 operational counter set is Load/Store/Ds/Km; gfx11 is Vm/Vs/Lgkm/Exp. `Exp` is a legacy ledger exception, not a portable gfx12 wait spelling. Architecture strings in the table are assembly, while Counter names are case-sensitive script enum variants.

```rip body
wg.isa().wait(if arch.gfx12() { Counter::Store } else { Counter::Vs }, 0)?;
```

## 12. Diagnostics and the agent loop

```sh
peacemaker-rip check kernel.rip --variant default
peacemaker-rip check kernel.rip --variant default --arch gfx1201 --json
peacemaker-rip fmt kernel.rip
peacemaker-rip fmt --check kernel.rip
peacemaker-rip explain R0301
peacemaker-rip explain --list
peacemaker-rip build kernel.rip both default kernel.s
peacemaker-rip build --certify kernel.rip gfx1151 default kernel.s
```

Read the code, severity, file/span and source caret, then the help and suggested replacement. JSON diagnostics include `code`, `severity`, `message`, `file`, `span`, `help`, `suggestion`, `notes`, and `explain`. Fix the earliest underlying error and rerun every architecture and variant you intend to ship. R0603 is an error: a missing effect can remove a necessary dependency wait. R0604 rejects a register read with no entry initialization or write anywhere in the lifted kernel; it is not an ordering/path proof. `check` permits any remaining warnings (exit 0); errors exit 1 and usage errors exit 2.

The [diagnostic reference](ERRORS.md) explains each stable code. R01xx is parsing/macros, R02xx evaluation, R03xx the driver, R04xx the builder, R05xx entry contract, R06xx instruction validation, R07xx certification and R09xx tools. `fmt` uses rustfmt with token-stream verification; formatting does not compile a kernel. `build` writes assembly; `both` produces architecture-suffixed output paths. `--certify` additionally assembles/links and runs wait-ledger replay and byte-exact lifting checks; it is not a numerical test.

### Source map

Entry/evaluation: `src/lib.rs`, `interp.rs`, `ops.rs`, `builtin.rs`, `expand.rs`, `fmt.rs`, `preprocess.rs`. Script-native signatures: `src/native.rs` (production generators: `src/native_gen.rs`). Kernel/layout and ownership: `peacemaker-isa/src/plan.rs`, `reg.rs`, `emit.rs`. Instruction helpers and waits: `peacemaker-isa/src/kernels/common.rs`, `ledger.rs`. Driver state/control flow: `peacemaker-author/src/runtime.rs`. Instruction spellings/operand widths: `peacemaker-ir/isa/gfx1100.tbl`, `gfx12.tbl`. ABI entry initialization: `peacemaker-ir/src/emu/mod.rs`.

## 13. Production generators: `iu4_gemm` and `qwen4_moe_sym`

The fast kernels are Rust generators. A script reaches them in two layers, neither of which adds typestate surface: whole-kernel natives that take the generator's `Spec`, and (for `iu4_gemm`) its phase blocks, which a script composes under the same checks as its own code. Production output does not change: a composition that mirrors `iu4_gemm::emit` re-emits its bytes.

### Coarse natives

The `Spec` enums are enum values, like a script's own unit enums: `Fold::{K128, K256Shared, K256Pow2}`, `Tile::{T128x128x8, T256x128x16}`, `Cacc::{One, Two}`, `Epi::{Set, Add, AddSigmoid, GateUpSilu, GateUpSiluBf16, QkvzaGdn}`, `ALayout::{Token, Slab}` (`iu4_gemm`) and `MoeKind::{GateUp, Down}`, `WeightCodec::{Mq4Sym, Mq3I4, E8pI4}` (`qwen4_moe_sym`; `MoeKind` is the Rust `qwen4_moe_sym::Kind`, renamed so it cannot clash with the `enum Kind` a script declares for its own variants). A script's own `enum Fold` (or any of these names) wins over the native one. `Arch` is the existing `Arch::Gfx1201`. The struct literals map one to one onto the Rust `Spec`; every field is required, and an unknown field is an error.

| native | Rust | result |
|---|---|---|
| `Iu4Spec { fold, tile, cacc, epi, act, arch }` | `iu4_gemm::Spec` | a spec value |
| `Iu4Gemm::emit(spec)` | `iu4_gemm::emit` | `Result<Emitted, String>` |
| `Iu4Gemm::emit_module(fold, tile, cacc, act, arch)` | `iu4_gemm::emit_module` | `Result<(Vec<Emitted>, String, String), String>`: the symbols, the module text and its proof as JSON |
| `MoeSymSpec { arch, kind, nt, rr, codec }` | `qwen4_moe_sym::Spec` | a spec value |
| `MoeSym::emit(spec)` | `qwen4_moe_sym::emit` | `Result<Emitted, String>` |
| `MoeSym::emit_module(arch)` | `qwen4_moe_sym::emit_module` | as `Iu4Gemm::emit_module` |
| `MoeSym::emit_module_codec(arch, codec)` | `qwen4_moe_sym::emit_module_codec` | as `Iu4Gemm::emit_module` |

`Spec::validate` stays the gate: a combination without a generator returns the Rust refusal (`iu4_gemm targets gfx1201 only`, ...).

```rip arch=gfx1201
fn emit(spec: Spec) -> Result<Emitted, String> {
    Iu4Gemm::emit(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::GateUpSilu, act: ALayout::Slab, arch: spec.arch,
    })
}
```

### `iu4_gemm` phase blocks

`Iu4Gen::new(Iu4Spec { .. })?` builds the generator of one valid spec (`Spec::validate`, gfx1201 only): the register assignment and geometry every block reads. It is an opaque value with read-only accessors.

| call | meaning |
|---|---|
| `gen.plan()?` | the kernel's `RegPlan` (a script may add ranges) |
| `gen.kernel_spec()` | the kernel's `KernelSpec` (symbol, kernargs, workgroup size) |
| `gen.spec()` | the `Iu4Spec` |
| `gen.reg("name")` | `Option<u8>`, the base of a physical register by name: `cacc acc magic d f0 f1 ds_nx sz_nx fr_a0 fr_a1 fr_w0 fr_w1 st_lds st_a meta_ds meta_sz ds_voff sz_voff sc_addr d_addr`, `a_pf<i> w_pf<i> st_w<i>`, SGPRs `karg srd_w srd_z srd_a0 srd_a1 srd_y step2 goff trips rs bs wave gpr136 mm1 hs tmp epi_s a_slab1`, kernel arguments `arg_a arg_u arg_gate arg_xq arg_y arg_m arg_k arg_n arg_bcx arg_bcy`; `None` for an unknown name or an absent optional register |

The blocks run inside a `Workgroup::new` scope. `lds` is the array of the four LDS rings (A, W, DS, SZ); `end` is a kernel exit from `wg.exit(..)`.

| call | Rust (`iu4_gemm::blocks`) | rings |
|---|---|---|
| `iu4::declare_lds(wg, gen)?` | `declare_lds` | declares them, every buffer `Free`; returns `lds` |
| `iu4::prologue(wg, gen, end, lds)?` | `prologue` | `Free`/`Free` in, `Published`/`Free` out (the block boundary); returns `lds` |
| `iu4::kloop(wg, gen, lds)?` | `kloop` | the whole rolled K-loop with its peeled group; returns `lds` |
| `iu4::block(wg, gen, tag, h, fetch_next, end_label, entry_label, lds)?` | `block` | one K128 block on plane `h` (0 or 1); `tag` names its labels, `end_label` closes its register lifetimes, `entry_label = false` leaves the block's own label to a `wg_skip_if` target; returns `lds` |
| `iu4::block_stepping(.., entry_label, step, lds)?` | `block_stepping` | `block` that also advances descriptor `step` (0 or 1) right after its slab-1 fetch: the production odd block does this for descriptor 0, inside one block, so it cannot sit between two `block` calls |
| `iu4::step_descriptor(wg, gen, which)?` | `step_descriptor` | `s_add_nc_u64` of the group stride onto descriptor 0 or 1 |
| `iu4::trip_end(wg, gen)?` | `trip_end` | odd descriptor, group offset and trip counter advance |
| `iu4::trips_cmp(wg, gen, "s_cmp_lg_u32")?` | `trips_cmp` | a `WgCond` for `wg_skip_if` / `loop_carried` (`s_cmp_{eq,lg,lt,le,gt,ge}_{u32,i32}` only) |
| `iu4::epilogue(wg, gen, end)?` | `epilogue` | the store path; places `iu4::EPI` |
| `iu4::gdn_epilogue(wg, gen, lds)?` | `gdn_epilogue` | the fused projection's GDN preparation; consumes `lds` (`Epi::QkvzaGdn` only) |
| `iu4::label(tag)`, `iu4::tail_label(h)` | `label`, `tail_label` | `.Liu4_<tag>`, `.Liu4_t<h>` |
| `iu4::ENTRY K_BEGIN K_LOOP K_LOOP_END EPI END GDN` | the label constants | strings |

Each block is one checked `Driver::typed` call. The native takes the four rings from the driver, runs the Rust block on the typed `Workgroup`, and gives the rings back to the same driver slots, so a block cannot read a `Free` buffer, and a script loop (`loop_carried`) or branch (`wg_skip_if`) around blocks is checked by the driver's ordinary rule that a body leaves every ring in the state it found it. What the bridge refuses:

- a ring of the wrong kind (the W ring where the A ring is expected: the tag differs), a ring in another phase, a ring with an undrained store, a ring the script made with `Ring::new` (only rings a typed block gave can be lent);
- any typed call under wave-uniform control (`skip_if`, `exec_if`, `if_else`, `forward`, `loop_until`), after a wave-uniform kernel exit, or on a target other than gfx1201 (the blocks are gfx1201-typed);
- a typed body that leaves a taken ring out. After any `Err` from a typed call the lowering is dead, like any backend `Err`: stop at the first one.

The fused per-WMMA fold stage stays inside `block`: it is interleaved within one instruction bundle. Exposing `fragments` / `bundle` / `publish_*` as separate natives is the next step if an experiment needs to change a block's inside; it uses the same bridge.

The committed twin [`iu4_gemm.rip`](../../peacemaker-isa/src/kernels/iu4_gemm.rip) composes `declare_lds`, `prologue`, a K-loop spelled out in script (`wg_skip_if` + `loop_carried` + `block` + `block_stepping` + `trip_end`, then the peeled pair) and `epilogue` / `gdn_epilogue`. Its variants are the shipped variant names (`k128-128x128x8-cacc1-slab-silu-bf16`; the grammar is in the file header). A test (`tests/iu4_gemm_twin.rs`) requires its output to be byte-identical to `iu4_gemm::emit` for every valid spec; [cookbook 11](cookbook/11-production-blocks.md) shows how to vary it.
