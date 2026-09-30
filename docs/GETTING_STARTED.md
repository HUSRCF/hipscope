# Getting started

Audience: first install on an AMD GPU host. Goal: install → verify → pull a model → run or chat.

## Prerequisites

- **Linux:** AMD GPU with `/dev/kfd` plus a ROCm HIP stack. A release-tag
  install on an admitted GPU (gfx1201, gfx1100, gfx1151, gfx906, gfx942) can
  use the tag's prebuilt [kernel pack](#prebuilt-kernel-packs) for the kernel
  registry, and installs with only the HIP runtime (`lib/libamdhip64.so`,
  `libhsa-runtime64.so`) and `bin/rocm_agent_enumerator`. Kernels outside the
  registry still JIT on first use — today that includes six modules of Qwen3.8
  H2 decode on gfx1201 — so running models still needs hipcc. The selected
  ROCm root should therefore also provide `include/hip/hip_runtime.h` and
  `bin/hipcc`. Install a supported AMD ROCm HIP runtime, development headers, and device
  compiler via
  [AMD's live install selector](https://rocm.docs.amd.com/en/latest/install/rocm.html)
  (choose packages for your GPU, OS, and ROCm version — package names drift;
  the selector is authoritative).
- **Supported ROCm range:** Linux with **ROCm 6 or newer** (project baseline from
  [README.md](../README.md)). **ROCm 6.4+** for RDNA4 (`gfx1200`/`gfx1201`);
  **ROCm 7.2+** for Strix Halo / gfx115x. hipfire's path resolver does not
  hardcode a required release — install a supported stack for your GPU.
- **Windows:** [AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) (`hipcc` + `amdhip64.dll`).
- **WSL2:** install AMD WSL GPU support first (`sudo amdgpu-install --usecase=wsl`), then use the Linux installer inside the distro.
- Disk space for models under `~/.hipfire/models/` (a few GB for small tags; tens of GB for 27B+).

Live model tags, VRAM floors, and formats: [MODELS.md](MODELS.md). Full env list: [env-vars.md](env-vars.md).

For a non-default or side-by-side install, pin one coherent SDK root before
starting hipfire:

```bash
export HIPFIRE_ROCM_PATH=/absolute/path/to/rocm
# If HIPFIRE_ROCM_PATH is unset, ROCM_PATH then HIP_PATH are accepted:
# export ROCM_PATH=/absolute/path/to/rocm
# export HIP_PATH=/absolute/path/to/rocm   # or .../hip (normalized to the parent)
```

Priority is `HIPFIRE_ROCM_PATH` > `ROCM_PATH` > `HIP_PATH`. An explicit override
is authoritative: once a root is selected, HIP/HSA libraries, headers, and
`hipcc` stay in that root family — hipfire will not fall back to another install
or a bare soname. If several complete roots are equally eligible (for example
multiple `/opt/rocm-*` with no active `/opt/rocm`), discovery refuses to guess;
set `HIPFIRE_ROCM_PATH` to one absolute root.

## Install

### Linux — master or beta in one command

The revision selector controls the managed source checkout under
`~/.hipfire/src`; no installer editing is needed:

```bash
# Current master:
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh | bash

# Integration/testing branch:
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh \
  | bash -s -- --branch beta
```

Branch installs remain on that branch when `hipfire update` is run without a
selector. The equivalent generic form is `--ref beta`; it auto-detects a
branch, tag, or commit. `HIPFIRE_INSTALL_REF=beta` is available for automation.

Both `master` and `beta` are mutable. For a reproducible install, pin and
inspect the installer itself, then ask it to install the same tag or commit:

```bash
PIN=v0.2.1
curl -fsSL "https://raw.githubusercontent.com/warpfront/hipfire/${PIN}/scripts/install.sh" \
  -o /tmp/hipfire-install.sh
sha256sum /tmp/hipfire-install.sh
less /tmp/hipfire-install.sh
bash /tmp/hipfire-install.sh --ref "$PIN"
```

Use `--tag v0.2.1` when the kind is known, or `--commit <full-sha>` for an
immutable commit. Fetching a pinned script but omitting the selector installs
`master`, so keep the two pins together.

The installer detects GPU arch and ROCm and builds the daemon and native CLI.
With `--tag` (or a tag-named `--ref`) on an admitted GPU it then installs the
tag's [prebuilt kernel pack](#prebuilt-kernel-packs); otherwise, or when the
pack cannot be used, it packages exact-source registry kernels with
`hipfire-kernel-pack`'s shared compiler. Either way the result is
`~/.hipfire/bin/kernels/compiled/<arch>/`, where each `.hsaco` requires a
matching `.index.json`; re-run the installer after upgrading an older install.
The bin directory can be added to `PATH`; reload the shell afterward.

GPUs not in the admitted registry remain JIT-only and need hipcc at runtime;
they cannot run with `HIPFIRE_NO_DEVICE_COMPILER=1`.

### Prebuilt kernel packs

Each release tag carries one pack per admitted architecture, built from the
tag's exact commit by `scripts/build-kernel-pack.sh` (hipcc, no GPU):

| Release asset | Contents |
|---|---|
| `hipfire-kernels-<tag>-<arch>.tar.gz` | `manifest.json` and `<arch>/` with every registry module's `.hsaco`, `.hash` and `.index.json` |
| `hipfire-kernels-<tag>-<arch>.tar.gz.sha256` | `sha256sum` line for the tarball |
| `hipfire-kernels-<tag>-<arch>.manifest.json` | copy of the manifest: commit, ROCm and HIP version, compiler identity, code-object version, admitted ROCm range |

`install.sh --tag <tag>` (also `--ref <tag>` and `hipfire update --tag <tag>`)
downloads the pack for the detected arch and installs it only when all of
these hold; otherwise it says why and compiles locally with hipcc:

- the tarball matches the published SHA-256, and holds only flat regular files;
- the manifest names the same tag, arch and commit as the checkout, and the
  runtime's kernel cache ABI;
- the local ROCm version (`<root>/.info/version`) is inside the manifest's
  range — by default the build's `major.minor` up to the next major release;
- a local hipcc, if any, is the same build that compiled the pack (a different
  one would make the runtime reject and recompile every kernel), and the
  active hipfire config selects the same kernel flags;
- every registry module's index matches the checked-out sources, recipe and
  object bytes, and the directory holds nothing else.

Pass `--compile-kernels` to skip the pack. `--kernel-pack-url` points at another
directory or URL holding the assets (`https://…`, `file:///…`, or a plain
path), for mirrors and offline installs:

```bash
bash install.sh --tag v0.4.0 --kernel-pack-url file:///srv/hipfire-packs/v0.4.0
```

`hipfire kernel-pack install --tag <tag> --arch <arch> --source ~/.hipfire/src`
runs the same download and checks on an existing install, for example to
restore kernels on a machine without hipcc.

Maintainers: the tag-triggered `release-kernel-packs` workflow
(`.github/workflows/release.yml`) builds every arch in a ROCm dev image and
attaches the assets to the release. The default image,
`rocm/dev-ubuntu-26.04:10.0.0-full`, carries the same ROCm 10.0.0 packages
(HIP 7.15.26333) as the project's GPU hosts, so its packs admit ROCm
[10.0, 11.0) and install on hosts with that hipcc. A pack only installs where
the local hipcc, if any, is the build that compiled it; hosts on another ROCm
compile locally. Repository variables choose another image
(`HIPFIRE_ROCM_IMAGE`) and, optionally, a wider admitted range
(`HIPFIRE_PACK_ROCM_MIN` and `HIPFIRE_PACK_ROCM_MAX_EXCLUSIVE`, set both). To
publish from a local toolchain instead:

```bash
scripts/build-kernel-pack.sh --tag v0.4.0 --out dist            # every admitted arch
scripts/build-kernel-pack.sh --tag v0.4.0 --out dist gfx1201    # or a subset
gh release upload v0.4.0 dist/hipfire-kernels-v0.4.0-*
```

The script compiles `git archive` of the tag's commit with a fresh `HOME` and
no `HIPFIRE_*` feature overrides, verifies every index against that toolchain,
and writes byte-reproducible tarballs. `--rocm-min X.Y --rocm-max-exclusive X.Y`
declares a range other than the default; it must contain the build's ROCm.

### Windows — select a branch, tag, or commit

```powershell
# Current master:
iex (irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1)

# Integration/testing branch:
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1))) `
  -Branch beta
```

For a reviewed, pinned installation:

```powershell
$Pin = "v0.2.1"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/warpfront/hipfire/$Pin/scripts/install.ps1" `
  -OutFile "$env:TEMP\hipfire-install.ps1"
Get-FileHash "$env:TEMP\hipfire-install.ps1" -Algorithm SHA256
notepad "$env:TEMP\hipfire-install.ps1"
& "$env:TEMP\hipfire-install.ps1" -Ref $Pin
```

PowerShell also accepts `-Branch beta`, `-Tag v0.2.1`, and `-Commit <sha>`.
The native `hipfire update` command remains Linux-only because Windows cannot
atomically replace the running executable; re-run `install.ps1` with the
desired selector instead.

Builds `daemon.exe` and the native CLI from the same selected source revision
under `~\.hipfire\src`. With `-Tag` (or a tag-named `-Ref`) on an admitted GPU
the installer runs `hipfire kernel-pack install` for the tag's
[prebuilt kernel pack](#prebuilt-kernel-packs) (`-KernelPackUrl` overrides the
download location, `-CompileKernels` skips it). Otherwise, or when the pack
cannot be used, hipcc from the HIP SDK packages exact registry sources: the
installer runs `daemon.exe --precompile` to produce indexed packages in
`~\.hipfire\bin\kernels\compiled\<arch>\`. Re-run `install.ps1` after
upgrading: copying bare `.hsaco` files is insufficient.

Unsupported GPU architectures likewise require hipcc JIT at runtime; the
installer does not copy bare checkout objects into the installed cache.

### Source checkout

```bash
git clone https://github.com/warpfront/hipfire
cd hipfire
cargo build --release --features deltanet --example daemon -p hipfire-runtime
cargo build --release -p hipfire-cli
cargo build --release -p hipfire-quantize
# optional TUI:
cargo build --release -p hipfire-tui
./scripts/install.sh   # from a checkout: local mode wires CLI + PATH
```

Other packaging: [NIXOS.md](NIXOS.md), [CONTAINER.md](CONTAINER.md).

## Uninstall a managed Linux install

The default uninstall removes the installed binaries, kernels, clean managed
source checkout, runtime PID/log files, and the PATH entry created by the
installer. It preserves downloaded models and settings under `~/.hipfire`:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh | bash
```

Preview without changing anything:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh \
  | bash -s -- --dry-run
```

Use `--purge` only when models, configuration, and every other file under
`~/.hipfire` should also be deleted. The script asks for an explicit
confirmation; automation can add `--yes`. It does not remove ROCm, Rust, or
other shared system dependencies.

## Verify

```bash
hipfire --version
hipfire version
hipfire diag
```

`--version` prints the release, source commit, and source ref in one line.
`hipfire version` additionally reports whether the managed checkout still
matches the binary and hashes the installed daemon; add `--json` for support
reports. `diag` reports GPU arch, VRAM, HIP/ROCm, kernel locations, model dir,
and config overrides.

### Update or switch channels on Linux

```bash
hipfire update                  # advance the currently selected branch
hipfire update @beta            # auto-detect branch/tag/commit
hipfire update --branch master
hipfire update --tag v0.2.1
hipfire update --commit <sha>
```

Tags and commits are detached, immutable pins; a later selector is required to
move away from them. Before switching, the updater retains the previous commit
under `refs/hipfire/backups/` and stashes dirty files with a recoverable
`hipfire-update-*` label.

## First inference

```bash
hipfire pull qwen3.5:4b
hipfire run  qwen3.5:4b "Explain FFT in one line"
```

- `pull` downloads the registry artifact into `~/.hipfire/models/` (and published sidecars when the registry entry lists them).
- `run` accepts a registry tag, alias, or path. **Recognized registry tags that are not local yet are auto-pulled** (can be multi-GB). Unresolved tags, aliases, or paths error with a `pull` / `list --remote` hint — they do not download.
- Cold start loads weights and may JIT kernels; later calls are faster if a daemon is already up.

Interactive multi-turn:

```bash
hipfire chat qwen3.5:4b
```

See [CHAT.md](CHAT.md).

## Keep a daemon warm

```bash
hipfire serve qwen3.5:4b -d    # background; OpenAI-compatible HTTP
hipfire run qwen3.5:4b "..."   # reuses serve when healthy
hipfire stop                   # graceful stop of the tracked daemon
```

Defaults (overridable in config): bind **`127.0.0.1:11435`** (loopback only), pre-warm **`default_model`** (`qwen3.5:9b` unless you set another). HTTP surface: [SERVE.md](SERVE.md). Subcommand flags: [CLI.md](CLI.md).

> **No auth / no TLS:** the serve HTTP API has **neither authentication nor TLS**. The default bind `127.0.0.1` accepts connections from this machine only. `hipfire config set host 0.0.0.0` (or `hipfire serve 0.0.0.0 11435`) listens on all interfaces and exposes inference to any reachable network (including chat-spawned serves). Expose beyond localhost only on a trusted/firewalled network **or** behind an **authenticated TLS-terminating reverse proxy** you control — never publish the raw port to the internet.

Force a one-shot daemon and skip HTTP:

```bash
HIPFIRE_LOCAL=1 hipfire run qwen3.5:4b "..."
```

## Light configuration

```bash
hipfire config                                      # global TUI → ~/.hipfire/config.toml
hipfire config qwen3.5:9b                           # resolved per-model policy
hipfire config qwen3.5:9b set generation.temperature 0.7
hipfire config qwen3.5:9b list                      # overlay + provenance
```

Defaults that matter on day one (from the native schema; full table in [CONFIG.md](CONFIG.md)). **Sampling send path:** `run` / `serve` transmit explicit request/CLI values, per-model TOML overlays, or the complete registry `recommended_settings` recipe (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty`, plus fallback `system_prompt`). Otherwise sampling fields are omitted for daemon/HFQ/arch fallback. Bare global sampling values alone are **not** effective `run`/`serve` defaults. **Chat** is the exception — it uses a global config snapshot for the session ([CHAT.md](CHAT.md)).

| Key | Default | Note |
|---|---|---|
| `temperature` | `0.3` | Stored global default only for run/serve send (see above); Chat session seed |
| `max_tokens` | `4096` | Per-request generation cap for `run` / API fallback |
| `kv_cache` | `auto` | Resolves via registry `default_kv_mode`, else `q8` |
| `dflash_mode` | **`off`** | DFlash is opt-in; pulling a draft does not enable it |
| `speculation` | `auto` | Mechanism selector; DFlash stays off when `dflash_mode=off`, but eligible **MTP / DSpark** paths may still activate under `auto`. Use `speculation=off` to force plain AR. |
| `thinking` | `on` | Reasoning models may emit `<think>`; display strip is CLI/API-side |
| `host` / `port` | `127.0.0.1` / `11435` | Serve bind (no auth, no TLS) |

Enable draft-model speculation only when you intend to:

```bash
hipfire pull qwen3.5:9b
hipfire pull qwen3.5:9b-draft
hipfire config set dflash_mode auto    # or on / per-model
```

`auto` uses a pulled draft when present; `on` fails the load without it.
`developer.dflash_draft` / `HIPFIRE_DFLASH_DRAFT` override the registry sidecar.
Several tags can share one draft file — `hipfire rm` keeps that sidecar while
another installed target still declares it (see [CLI.md](CLI.md)).

## Long context (optional)

CASK/TriAttention eviction is experimental and disabled by default. It is
**not** required for short prompts. When deliberately testing it for long
context on limited VRAM:

1. Prefer models whose `pull` ships a `.triattn.bin` sidecar, **or** generate one: `hipfire sidecar-gen <model>`.
2. Set `cask_sidecar` to that exact path. Enable `cask` separately only when m-folding is intended.
3. Read constraints (A3B, DFlash + m-fold) in [CONFIG.md](CONFIG.md) before enabling on MoE or with DFlash. `cask_auto_attach` remains `false` unless explicitly opted in.

### Measured capacity (Qwen3.5/3.6 35B-A3B-class, 24GB)

On 24GB GPUs, Q8 KV is about 10,880 B/token across 10 full-attention layers,
plus O(N) flash partials (~2,064 B/token), ~25 MiB DeltaNet state, and multi-GiB
fixed HIP/graph overhead. **50K Q8 is tight but physically feasible**; **200K Q8
is not a 24GB-class configuration** — it needs >32GB-class VRAM or CASK/compressed
KV. Some historical 131K sidecar benches clamped physical capacity to ~2432
tokens and should not be read as full-context Q8 support. Long-context decode
slowdown is expected O(N) full-attention bandwidth, not by itself an admission
regression.

## If something fails

| Symptom | What to try |
|---|---|
| `hipfire: command not found` | Reload shell; ensure install dir is on `PATH` |
| HIP / `/dev/kfd` / arch errors | `hipfire diag`; match ROCm/HIP version to arch (above) |
| Model not found | `hipfire list` / `hipfire list -r`; `hipfire pull <tag>` |
| Port in use / stale serve | `hipfire ps`; `hipfire stop --force`; check `~/.hipfire/serve.pid` and (detached only) `serve.log` |
| Draft pulled but no speedup | Expected: `dflash_mode` defaults to **off** |
| Truncated answers on thinking models | Raise `max_tokens` / `thinking_budget`; see [MODELS.md](MODELS.md) thinking section |

```bash
hipfire diag
tail -f ~/.hipfire/serve.log
```

## What to read next

| Doc | When |
|---|---|
| [CLI.md](CLI.md) | Every subcommand and flag surface |
| [CHAT.md](CHAT.md) | Interactive chat, thinking display, daemon attach |
| [SERVE.md](SERVE.md) | OpenAI-compatible HTTP |
| [MODELS.md](MODELS.md) | Tags, VRAM, BYO quantize, thinking/templates |
| [CONFIG.md](CONFIG.md) | All config keys and experimental CASK/TriAttention opt-ins |
| [QUANTIZE.md](QUANTIZE.md) | `hipfire quantize` operator guide |
| [INDEX.md](INDEX.md) | Ownership map for the rest of `docs/` |
