# 17. Fine K-loop natives: place the waits, publishes and fetches of an `iu4_gemm` block yourself

[Chapter 11](11-production-blocks.md) composes `iu4_gemm` from whole blocks: `iu4::block` is one opaque native, so where a block's slab fetches, LDS publishes, waits and WMMA groups sit is not reachable from a script. This chapter splits the block. The dense K128 block is a composition of finer typed natives that the generator itself is written over, and a script can spell the block out with them, in the generator's order (the committed twin does, byte for byte) or in another one. The driver's typing is unchanged: a schedule that publishes into a buffer that is not `Free`, rotates before the wait, or reads or WMMAs a slot nobody published is a build error, not a wrong answer on the GPU.

The reference table of the natives is [reference §13](../REFERENCE.md); this chapter explains how they compose.

## One block, step by step

A K128 block on plane `h` is two 64-K slabs. Slab 0 sits in LDS slot 0 of the payload rings (A: activations, W: weights); slab 1 was fetched by the preceding block and is published into slot 1 while slab 0's WMMAs run. After the rendezvous the block reads slot 1, folds, and stages the next block's slab 0 into slot 0.

```text
block_begin                    label, register lifetimes
[step_descriptor]              block_stepping's descriptor advance
-- phase A (slab 0, slot 0) --
fragments                      LDS -> registers, both 32-K sub-blocks
wmma sb0 (seed)    wmma sb1    the two WMMA groups
publish_slab                   slab 1 -> slot 1: Free -> Writing      } WAIT places it
fetch_next_meta                next token scale / row header
wait "slab"                    Writing pending -> drained
barrier rotate(A), rotate(W)   slot 0 retired, slot 1 published
-- phase B (slab 1, slot 1) --
fetch_next                     next block's slab 0 -> prefetch registers
fragments  d_loads             slot 1; token scales
wmma sb0           scale_loads           (label _s0)
wmma sb1 (fold)    scale_tail            (label _s1)
publish_slab  publish_meta     next block -> slot 0 and the next metadata plane
wait "all"
signal rotate(A, W, DS, SZ)    split rendezvous
fetch_slab1 | fetch_slab1_next_group     next block's slab 1 -> prefetch registers
fold_scale                     the fold's stage 2
wait_arrived
```

The peeled last block of the K-loop (`fetch_next = false`) has no next block: no `fetch_next*`, no second publish, no rendezvous, and its phase B ends with `fold_scale`.

## The natives

| call | what it emits |
|---|---|
| `iu4::block_begin(wg, g, tag, h, end_label, entry_label, lds)?` | the block's label and register lifetimes |
| `iu4::fetch_slab1(wg, g, h)?`, `iu4::fetch_slab1_next_group(wg, g)?`, `iu4::fetch_next(wg, g, h)?`, `iu4::fetch_next_meta(wg, g, h)?` | the global loads into the prefetch registers (slab 1 of block `h`; the odd block's form for the next group; the next block's slab 0; its token scale and row header) |
| `iu4::publish_slab(wg, g, lds)?`, `iu4::publish_meta(wg, g, lds)?` | the LDS stores of the staged slab, or of the next metadata plane, into the ring's `Free` next buffer |
| `iu4::wait(wg, lds, set)?` | the counter wait for `"slab"`, `"meta"` or `"all"` rings; returns the `Drained` tokens |
| `iu4::fragments(wg, g, lds)?`, `iu4::d_loads(wg, g, lds)?`, `iu4::scale_loads(wg, g, tag, lds)?`, `iu4::scale_tail(wg, g, tag, lds)?` | LDS reads of the `Published` current buffers |
| `iu4::wmma(wg, g, sb, first, fold, lds)?` | the eight WMMAs of 32-K sub-block `sb`; `first` seeds the accumulators; `fold` interleaves the fold's stage 1 |
| `iu4::fold_scale(wg, g)?` | the fold's stage 2 |
| `iu4::step_descriptor(wg, g, which)?` | the descriptor advance |

Ring-touching natives take `lds` (the array of the four rings) and return it. The rendezvous is the generic driver's `wg.barrier((rotate(lds[0], da), ..))` or the split `wg.signal(..)` / `wg.wait_arrived(..)`; `iu4::wait` returns the tokens `rotate` takes (`rotate(ring)` alone also works: the driver checks the ring's own state, not the token).

## Typing rules

A ring crosses between a native and the driver in the state the Rust step is typed for (`Lend::take` for `Free`/`Published` buffers, `Lend::take_writing` for a `Writing` next buffer, whose `Pending` token travels with it). So:

| rule | refused as | test (`tests/iu4_gemm_fine_twin.rs`) |
|---|---|---|
| a publish into a buffer that is not `Free` | R0399 | `publish_into_a_non_free_buffer_is_refused` |
| a rotate before the wait | R0301 | `rotating_before_the_wait_is_refused` |
| a wait with nothing in flight, or an unknown ring set | R0399 | `waiting_without_a_publish_is_refused` |
| a fragment/scale read or WMMA of an unpublished slot | R0303 | `reading_or_wmma_of_an_unpublished_slot_is_refused` |
| a ring of the wrong kind, or one the script made | R0399 | `a_ring_of_the_wrong_kind_is_refused_by_the_fine_natives`, `a_ring_the_script_made_cannot_enter_the_fine_natives` |
| a fine step under wave-uniform control | R0304 | `a_fine_step_under_wave_uniform_control_is_refused` |
| a loop trip that ends with a publish in flight | R0399 | `a_trip_that_leaves_a_publish_in_flight_is_refused` |

Each has a test in `tests/iu4_gemm_fine_twin.rs`, and the four below a whole failing script ([ERRORS.md](../ERRORS.md) points here from R0301, R0303 and R0399). The ring tags (`PayA`, `PayW`, `MetaD`, `MetaS`) are the typed lend's, so `iu4::publish_slab` with the W ring where the A ring belongs is refused. The loop rule is the driver's: `loop_carried` demands every ring leave the trip in the state it entered, which is why a block that publishes also waits and rotates inside the trip.

`fetch_*`, `fold_scale`, `wmma` and `step_descriptor` emit through the builder rather than `Driver::typed` (which the driver refuses while a split barrier is in flight), so they may sit between `signal` and `wait_arrived`; `wmma` is gated on the payload rings' phases instead (the current buffer must be `Published`).

## What the driver refuses

Four of the rules, as whole scripts (each the same two-function `iu4_gemm` kernel with one offending call; `tests/docs.rs` compiles each and requires its code). They are gfx1201-only, like every `iu4_gemm` script.

Publish into a buffer that is not `Free`: the second `publish_slab` finds the ring's next buffer `Writing`, its stores not yet waited and rotated.

```rip arch=gfx1201 fails=R0399
fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}
fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    let lds = iu4::publish_slab(wg, g, lds)?;
    // The slab above is still Writing: its stores were never waited and rotated.
    let lds = iu4::publish_slab(wg, g, lds)?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}
```

Fix: `iu4::wait(wg, lds, "slab")?` and a `wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?` before the next publish, as `fine_block` does.

A wait with nothing in flight:

```rip arch=gfx1201 fails=R0399
fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}
fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    // Nothing was published into the next buffers: there is no store to wait for.
    let _drained = iu4::wait(wg, lds, "slab")?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}
```

Fix: wait only after the publish whose stores it drains.

A rotate before the wait: `s_barrier` does not wait for memory counters, so the driver refuses to publish the slab's undrained stores.

```rip arch=gfx1201 fails=R0301
fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}
fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    let lds = iu4::publish_slab(wg, g, lds)?;
    // The slab's stores are still pending.
    wg.barrier((rotate(lds[0]), rotate(lds[1])))?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}
```

Fix: drain first and hand the tokens to the rotate: `let (da, dw) = iu4::wait(wg, lds, "slab")?;` then `wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?;`.

A WMMA group (or a fragment or scale read) of a slab nobody published: the rings are still `Free` as `declare_lds` made them.

```rip arch=gfx1201 fails=R0303
fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}
fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    // No block has been published: the first WMMA group has no slot to read.
    let lds = iu4::wmma(wg, g, 0, true, false, lds)?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}
```

Fix: run the prologue first (it stages and publishes block 0), then `iu4::fragments(wg, g, lds)?` and the WMMA groups, as `fine_block` does.

## The twin composition

`iu4::block` is `kloop::block_with`, and `block_with` is the sequence above written in Rust over the same functions the natives call. [`iu4_gemm_fine.rip`](../../../peacemaker-isa/src/kernels/iu4_gemm_fine.rip) is the committed twin of `iu4_gemm.rip` with `mma_block` calling a script function `fine_block` (the block above, only fine natives) for the dense variants. `tests/iu4_gemm_fine_twin.rs` requires it to emit, for every dense variant, the bytes and proof of `iu4_gemm::emit`, so `iu4::block` and the spelled-out block are provably equal. The sparse variants keep the opaque sparse blocks.

## A wait-moved block

`const WAIT` in the twin picks where the next slab's publish sits between the WMMA groups: `"after"` is the shipped order; `"mid"` puts it between the two 32-K groups of each phase, `"before"` ahead of both. The stores then have more WMMA work between them and the wait, which is the knob the generator exposes as `Tune { wait: WaitPlace::MidMma | BeforeMma }`; the test requires `WAIT = "mid"` and `"before"` to emit exactly those generator bytes (`a_wait_moved_script_equals_the_generators_tune`). The same block as a standalone, runnable `K == 256` kernel, with `WAIT = "mid"`:

```rip arch=gfx1201
const WAIT: &str = "mid";

fn emit(spec: Spec) -> Result<Emitted, String> {
    let g = Iu4Gen::new(Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch, mma: Mma::Dense,
    })?;
    let mut b = Builder::new(g.kernel_spec(), g.plan()?);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    author(&mut wg, &g)?;
    b.finish()
}

fn author<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen) -> Result<(), String> {
    let lds = iu4::declare_lds(wg, g)?;
    let end = wg.exit(iu4::END)?;
    op(wg.isa(), "s_setprio 1", vec![], vec![])?;
    let lds = iu4::prologue(wg, g, end, lds)?;
    // K == 256 only: the peeled pair is the whole K-loop.
    wg.label(iu4::K_BEGIN)?;
    let lds = fine_block(wg, g, "t0", 0, true, iu4::label("t1"), true, None, lds)?;
    fine_block(wg, g, "t1", 1, false, iu4::EPI, true, None, lds)?;
    iu4::epilogue(wg, g, end)?;
    wg.end(end)
}

// One K128 block of the dense K-loop on plane `h`, written with the fine
// natives in the order the generator's `block` uses (`WAIT = "after"`; the other
// two placements are the generator's `Tune { wait: Mid | Before }`). `step`
// advances that descriptor right after the block's head, as `block_stepping`.
//
// Ring discipline, each line checked by the driver:
//   publish_*   next buffer Free -> Writing (a Pending token)
//   wait        Pending -> Drained (names the ring set)
//   barrier     rotate: cur retired, the Drained next one published
//   fragments / scale_loads / wmma   read the Published current buffer
fn fine_block<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen, tag: &str, h: usize, fetch_next: bool,
        end_label: &str, entry_label: bool, step: Option<usize>, lds: Lds) -> Result<Lds, String> {
    let lds = iu4::block_begin(wg, g, tag, h, end_label, entry_label, lds)?;
    if let Some(s) = step {
        iu4::step_descriptor(wg, g, s)?;
    }
    // Phase A: slab 0 from slot 0; slab 1 (fetched by the preceding block) is
    // published into slot 1.
    let lds = iu4::fragments(wg, g, lds)?;
    let lds = if WAIT == "after" {
        let lds = iu4::wmma(wg, g, 0, true, false, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, false, lds)?;
        iu4::publish_slab(wg, g, lds)?
    } else if WAIT == "mid" {
        let lds = iu4::wmma(wg, g, 0, true, false, lds)?;
        let lds = iu4::publish_slab(wg, g, lds)?;
        iu4::wmma(wg, g, 1, false, false, lds)?
    } else {
        let lds = iu4::publish_slab(wg, g, lds)?;
        let lds = iu4::wmma(wg, g, 0, true, false, lds)?;
        iu4::wmma(wg, g, 1, false, false, lds)?
    };
    if fetch_next {
        iu4::fetch_next_meta(wg, g, h)?;
    }
    let (da, dw) = iu4::wait(wg, lds, "slab")?;
    wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?;
    // Phase B: slab 1 from slot 1, the fold, the next block into slot 0.
    if fetch_next {
        iu4::fetch_next(wg, g, h)?;
    }
    let lds = iu4::fragments(wg, g, lds)?;
    let lds = iu4::d_loads(wg, g, lds)?;
    if !fetch_next {
        // The peeled group's last block: nothing to publish.
        let lds = iu4::wmma(wg, g, 0, false, false, lds)?;
        let lds = iu4::scale_loads(wg, g, tag, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, true, lds)?;
        let lds = iu4::scale_tail(wg, g, tag, lds)?;
        iu4::fold_scale(wg, g)?;
        return Ok(lds);
    }
    let lds = if WAIT == "after" {
        let lds = iu4::wmma(wg, g, 0, false, false, lds)?;
        let lds = iu4::scale_loads(wg, g, tag, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, true, lds)?;
        let lds = iu4::scale_tail(wg, g, tag, lds)?;
        let lds = iu4::publish_slab(wg, g, lds)?;
        iu4::publish_meta(wg, g, lds)?
    } else if WAIT == "mid" {
        let lds = iu4::wmma(wg, g, 0, false, false, lds)?;
        let lds = iu4::scale_loads(wg, g, tag, lds)?;
        let lds = iu4::publish_slab(wg, g, lds)?;
        let lds = iu4::publish_meta(wg, g, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, true, lds)?;
        iu4::scale_tail(wg, g, tag, lds)?
    } else {
        let lds = iu4::publish_slab(wg, g, lds)?;
        let lds = iu4::publish_meta(wg, g, lds)?;
        let lds = iu4::wmma(wg, g, 0, false, false, lds)?;
        let lds = iu4::scale_loads(wg, g, tag, lds)?;
        let lds = iu4::wmma(wg, g, 1, false, true, lds)?;
        iu4::scale_tail(wg, g, tag, lds)?
    };
    let (da, dw, dd, ds) = iu4::wait(wg, lds, "all")?;
    let b2 = wg.signal((rotate(lds[0], da), rotate(lds[1], dw), rotate(lds[2], dd), rotate(lds[3], ds)))?;
    // The signal drained slab-0 stores: the retired payload ring can hold the
    // next K128's slab 1 while this block folds. The odd block runs before the
    // group offset advances, so it names the next group explicitly.
    if h == 0 {
        iu4::fetch_slab1(wg, g, 1)?;
    } else {
        iu4::fetch_slab1_next_group(wg, g)?;
    }
    iu4::fold_scale(wg, g)?;
    wg.wait_arrived(b2)?;
    Ok(lds)
}
```

Move any step the same way: put `iu4::fetch_next_meta` before the first WMMA group, split the publish across the groups, delay `iu4::fold_scale` past the `wait_arrived`. A move that breaks the ring discipline fails at the offending native; a move that is legal builds, is certified like any script, and its result bits are the shipped ones (a reorder of independent steps changes only the schedule; the loads read the same published slots).

## Timing a variation

`pm_loop --kernel iu4_gemm ... --rip FILE` takes the full twin, not the standalone kernel above: `crates/peacemaker-isa/src/kernels/iu4_gemm_fine.rip` is the file to copy. For the 27B gate_up shape (M = 17408, N = 8192, K = 5120):

```sh
pm_loop --kernel iu4_gemm --arch gfx1201 --shape M=17408,N=8192,K=5120 \
        --fold k128 --tile 128x128x8 --cacc 1 --epi silu --alayout slab \
        --rip crates/peacemaker-isa/src/kernels/iu4_gemm_fine.rip
```

(`pm_loop` refuses, exit 3 stage `rip`, a file whose symbol, kernarg layout, workgroup size or LDS differs from the generator's; the fine twin keeps all of them.) Change `WAIT` or the body of `fine_block` and rerun: no rebuild.

## Ring depth 3

Not delivered. The two-buffer ring is not an accident of one type: it is the shape of the typed core, the runtime driver and the iu4 kernel at once, and a three-slot ring is a change in each.

- **Typed ring** (`peacemaker-author/src/lds.rs`): `Ring<R, C, N>` holds `bufs: [Slots; 2]`, roles `cur` and `next`, and `next_index()` is `1 - cur`; `Rotate`, `Prime`, `RetireCur` and the `StoreTarget` impls swap those two. A depth-D ring has `cur`, D-2 buffers published ahead, and a `next`: the phase of the middle buffers is a third type parameter (or a runtime vector the type does not see), and `Rotate` becomes `cur = (cur + 1) % D` with the oldest buffer retired.
- **Runtime driver** (`peacemaker-author/src/runtime.rs`): `RingBuf.bufs: [Vec<usize>; 2]`, `next = 1 - cur`, `Driver::ring(name, first, second)`, and `Lend::take`/`give`/`view` carry exactly `(cur, next)` phases. They need the same generalisation, and `Ring::new` in a script a three-region form.
- **The kernel**: `Layout` (`spec.rs`) is `[u32; 2]` per payload ring, and every fragment load and publish store carries its slot as an *immediate* (`kloop.rs` `fragments`, `publish.rs` `publish_slab`). A K-loop trip is four slabs (two K128 blocks, two slabs each); with three slots the slot of slab `k` is `k mod 3`, so the immediates repeat every twelve slabs: either unroll three trips (3x code) or keep the slot base in a register and add it per access (extra VALU on the critical path, and `ds_load` offsets stop being constants). LDS is not the limit (a third A+W slab buffer is +8192 bytes on the 128x128x8 tile: 26624 of 65536).
- **Prefetch**: depth 3 only helps if a slab is fetched two ahead, and the generator has one prefetch set (`a_pf` / `w_pf`). A second set is another 8+8 VGPRs against the same register plan.

Delivering it means the typed ring and driver generalisation first (contained to `peacemaker-author`, tests in `tests/ui`), then a depth-parameterised `Layout` and a rotating-base or unrolled K-loop in `iu4_gemm`, then the fine natives take the slot as an argument. The fine natives of this chapter already take the ring, not a slot number, so none of their signatures needs to change for it.
