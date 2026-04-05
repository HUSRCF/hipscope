# Windows Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make hipfire installable and runnable on Windows with AMD GPUs, including auto-download of the HIP runtime DLL.

**Architecture:** Platform-branch the HIP dlopen in `hip-bridge/ffi.rs` to load `amdhip64.dll` on Windows. Add a PowerShell installer (`scripts/install.ps1`) that detects GPU, downloads the DLL, installs Bun, clones/builds, and sets up PATH. Cross-compile the Rust binary from Linux using `x86_64-pc-windows-gnu`. Pre-compiled .hsaco kernels are platform-independent and work as-is.

**Tech Stack:** Rust (cross-compile), PowerShell, Bun (Windows), HIP runtime DLL

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/hip-bridge/src/ffi.rs` | Modify (~line 100-111) | Platform-branch dlopen for Windows vs Linux |
| `scripts/install.ps1` | Create | PowerShell Windows installer |
| `scripts/hipfire.cmd` | Create | Windows CLI wrapper batch file |
| `README.md` | Modify | Add Windows install instructions |

---

### Task 1: Platform-branch HIP dlopen

**Files:**
- Modify: `crates/hip-bridge/src/ffi.rs:100-111`

- [ ] **Step 1: Add platform-aware library loading**

Replace the current hardcoded `libamdhip64.so` load in `HipRuntime::load()` with a `cfg` branch:

```rust
impl HipRuntime {
    /// Load the HIP runtime via dlopen.
    /// Linux: libamdhip64.so from /opt/rocm/lib or system path.
    /// Windows: amdhip64.dll from %USERPROFILE%\.hipfire\runtime\, %HIP_PATH%\bin\, or system PATH.
    pub fn load() -> HipResult<Self> {
        let lib = unsafe {
            #[cfg(target_os = "windows")]
            let lib_result = {
                // Try hipfire's own runtime dir first, then HIP SDK, then system PATH
                let home = std::env::var("USERPROFILE").unwrap_or_default();
                let hipfire_dll = format!("{}\\.hipfire\\runtime\\amdhip64.dll", home);
                let hip_path_dll = std::env::var("HIP_PATH")
                    .map(|p| format!("{}\\bin\\amdhip64.dll", p))
                    .unwrap_or_default();

                Library::new(&hipfire_dll)
                    .or_else(|_| Library::new(&hip_path_dll))
                    .or_else(|_| Library::new("amdhip64.dll"))
                    .map_err(|e| {
                        HipError::new(
                            0,
                            &format!(
                                "failed to load amdhip64.dll: {e}.\n\
                                 Searched:\n  {}\n  {}\n  system PATH\n\
                                 Re-run the installer or install AMD HIP SDK.",
                                hipfire_dll, hip_path_dll
                            ),
                        )
                    })
            };

            #[cfg(not(target_os = "windows"))]
            let lib_result = Library::new("libamdhip64.so").map_err(|e| {
                HipError::new(
                    0,
                    &format!("failed to dlopen libamdhip64.so: {e}. Is ROCm installed?"),
                )
            });

            lib_result?
        };
```

This replaces lines 103-111 of the current `ffi.rs`. The rest of the function (symbol loading) is unchanged.

- [ ] **Step 2: Verify Linux build still works**

Run: `cargo build --release --features deltanet --example daemon -p engine 2>&1 | tail -3`
Expected: `Finished release profile` with no errors.

- [ ] **Step 3: Verify a quick smoke test**

Run: `timeout 15 target/release/examples/infer models/qwen3.5-0.8b.q4.hfq --no-think "Hello" 2>&1 | grep "Done"`
Expected: `=== Done: NN tokens in NNms (NNN.N tok/s) ===`

- [ ] **Step 4: Commit**

```bash
git add crates/hip-bridge/src/ffi.rs
git commit -m "hip-bridge: platform-branch dlopen for Windows amdhip64.dll"
```

---

### Task 2: Cross-compile for Windows

- [ ] **Step 1: Install the Windows cross-compile target**

```bash
rustup target add x86_64-pc-windows-gnu
```

Note: we use `gnu` (MinGW) not `msvc` because `msvc` requires the Windows SDK which isn't available on Linux. The `gnu` target produces a standard .exe that works on Windows.

- [ ] **Step 2: Install the MinGW cross-compiler**

```bash
sudo apt install -y gcc-mingw-w64-x86-64
```

- [ ] **Step 3: Cross-compile the engine**

```bash
cargo build --release --target x86_64-pc-windows-gnu --features deltanet \
  --example daemon --example infer -p engine 2>&1 | tail -5
```

Expected: `Finished release profile` — the binary lands at `target/x86_64-pc-windows-gnu/release/examples/daemon.exe`.

If this fails with linker errors, create `.cargo/config.toml` with:
```toml
[target.x86_64-pc-windows-gnu]
linker = "x86_64-w64-mingw32-gcc"
```

- [ ] **Step 4: Verify the .exe exists**

```bash
file target/x86_64-pc-windows-gnu/release/examples/daemon.exe
```

Expected: `PE32+ executable (console) x86-64, for MS Windows`

- [ ] **Step 5: Commit config if created**

```bash
git add .cargo/config.toml 2>/dev/null
git commit -m "build: add x86_64-pc-windows-gnu cross-compile target" --allow-empty
```

---

### Task 3: PowerShell installer

**Files:**
- Create: `scripts/install.ps1`

- [ ] **Step 1: Write the PowerShell installer**

```powershell
# hipfire installer for Windows — detects GPU, installs deps, downloads binary + kernels.
# Usage: irm https://raw.githubusercontent.com/Kaden-Schutt/hipfire/alpha-builds/scripts/install.ps1 | iex
$ErrorActionPreference = "Stop"

$HipfireDir = "$env:USERPROFILE\.hipfire"
$BinDir = "$HipfireDir\bin"
$RuntimeDir = "$HipfireDir\runtime"
$ModelsDir = "$HipfireDir\models"
$SrcDir = "$HipfireDir\src"
$GithubRepo = "Kaden-Schutt/hipfire"
$GithubBranch = "alpha-builds"

Write-Host "=== hipfire installer (Windows) ===" -ForegroundColor Cyan
Write-Host ""

# ─── GPU Detection ───────────────────────────────────────
Write-Host "Checking for AMD GPU..."
$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" } | Select-Object -First 1
if (-not $gpu) {
    Write-Host "  ERROR: No AMD GPU detected." -ForegroundColor Red
    Write-Host "  hipfire requires an AMD Radeon GPU (RDNA1+)."
    Write-Host "  Check Device Manager > Display adapters."
    exit 1
}
Write-Host "  GPU: $($gpu.Name) ✓" -ForegroundColor Green

# Detect arch from device name (best-effort)
$GpuArch = "unknown"
switch -Regex ($gpu.Name) {
    "5700|5600"        { $GpuArch = "gfx1010" }
    "6[789]00|6600"    { $GpuArch = "gfx1030" }
    "7[789]00|7600"    { $GpuArch = "gfx1100" }
    "9[07]70"          { $GpuArch = "gfx1200" }
}
if ($GpuArch -eq "unknown") {
    Write-Host "  WARNING: Could not detect GPU architecture from name." -ForegroundColor Yellow
    Write-Host "  Supported: gfx1010 (5700 XT), gfx1030 (6800 XT), gfx1100 (7900 XTX), gfx1200 (9070)"
    $GpuArch = Read-Host "  Enter your GPU arch (or press Enter to skip)"
    if (-not $GpuArch) { $GpuArch = "unknown" }
}
Write-Host "  GPU arch: $GpuArch"

# ─── Create directories ─────────────────────────────────
New-Item -ItemType Directory -Force -Path $BinDir, $RuntimeDir, $ModelsDir | Out-Null

# ─── HIP Runtime DLL ─────────────────────────────────────
Write-Host ""
Write-Host "Checking HIP runtime..."
$dllPaths = @(
    "$RuntimeDir\amdhip64.dll",
    "$env:HIP_PATH\bin\amdhip64.dll",
    "C:\Program Files\AMD\ROCm\bin\amdhip64.dll"
)
$dllFound = $false
foreach ($p in $dllPaths) {
    if (Test-Path $p) {
        Write-Host "  amdhip64.dll: found at $p ✓" -ForegroundColor Green
        $dllFound = $true
        break
    }
}

if (-not $dllFound) {
    Write-Host "  amdhip64.dll: NOT FOUND" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Downloading HIP runtime DLL..."
    # AMD HIP SDK redistributable — just the runtime DLL
    # The DLL is extracted from the HIP SDK installer package
    $dllUrl = "https://github.com/$GithubRepo/releases/download/hip-runtime/amdhip64.dll"
    $dllDest = "$RuntimeDir\amdhip64.dll"
    try {
        Invoke-WebRequest -Uri $dllUrl -OutFile $dllDest -UseBasicParsing
        Write-Host "  Downloaded to $dllDest ✓" -ForegroundColor Green
    } catch {
        Write-Host "  Download failed: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Manual install options:" -ForegroundColor Yellow
        Write-Host "    1. Install AMD HIP SDK: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"
        Write-Host "    2. Copy amdhip64.dll to $RuntimeDir"
        Write-Host ""
        $continue = Read-Host "  Continue without HIP runtime? (hipfire won't run) [y/N]"
        if ($continue -ne "y") { exit 1 }
    }
}

# ─── Install Bun ─────────────────────────────────────────
Write-Host ""
if (Get-Command bun -ErrorAction SilentlyContinue) {
    Write-Host "Bun: found ✓" -ForegroundColor Green
} else {
    Write-Host "Installing Bun (runtime for hipfire CLI)..."
    try {
        powershell -c "irm bun.sh/install.ps1 | iex"
        # Refresh PATH for current session
        $env:PATH = "$env:USERPROFILE\.bun\bin;$env:PATH"
        Write-Host "  Bun installed ✓" -ForegroundColor Green
    } catch {
        Write-Host "  Bun install failed. Visit https://bun.sh" -ForegroundColor Red
        exit 1
    }
}

# ─── Clone repository ────────────────────────────────────
Write-Host ""
if (-not (Test-Path "$SrcDir\.git")) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "  ERROR: git is required. Install from https://git-scm.com" -ForegroundColor Red
        exit 1
    }
    Write-Host "Cloning repository..."
    git clone --depth 1 --branch $GithubBranch "https://github.com/$GithubRepo.git" $SrcDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Clone failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Cloned ✓" -ForegroundColor Green
} else {
    Write-Host "Existing clone found at $SrcDir"
    git -C $SrcDir pull origin $GithubBranch 2>$null
}

# ─── Install binaries ────────────────────────────────────
Write-Host ""
Write-Host "Installing hipfire..."

# Check for pre-built Windows binaries
$daemonExe = "$SrcDir\target\x86_64-pc-windows-gnu\release\examples\daemon.exe"
$daemonRelease = "$SrcDir\target\release\examples\daemon.exe"
if (Test-Path $daemonExe) {
    Copy-Item $daemonExe "$BinDir\daemon.exe" -Force
    Write-Host "  Pre-built Windows binary ✓" -ForegroundColor Green
} elseif (Test-Path $daemonRelease) {
    Copy-Item $daemonRelease "$BinDir\daemon.exe" -Force
    Write-Host "  Pre-built binary ✓" -ForegroundColor Green
} else {
    Write-Host "  No pre-built binaries. Building from source..."
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Host "  Installing Rust..."
        Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "$env:TEMP\rustup-init.exe"
        & "$env:TEMP\rustup-init.exe" -y --default-toolchain stable 2>$null
        $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    }
    Push-Location $SrcDir
    Write-Host "  cargo build --release (this may take several minutes)..."
    cargo build --release --features deltanet --example daemon --example infer -p engine
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  BUILD FAILED." -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Copy-Item "target\release\examples\daemon.exe" "$BinDir\daemon.exe" -Force
    Pop-Location
    Write-Host "  Built and installed ✓" -ForegroundColor Green
}

# Copy CLI
New-Item -ItemType Directory -Force -Path "$HipfireDir\cli" | Out-Null
Copy-Item "$SrcDir\cli\index.ts" "$HipfireDir\cli\index.ts" -Force
Copy-Item "$SrcDir\cli\package.json" "$HipfireDir\cli\package.json" -Force

# Create hipfire.cmd wrapper
@"
@echo off
bun run "%USERPROFILE%\.hipfire\cli\index.ts" %*
"@ | Set-Content "$BinDir\hipfire.cmd" -Encoding ASCII
Write-Host "  CLI installed to $BinDir ✓" -ForegroundColor Green

# ─── Install kernels ─────────────────────────────────────
Write-Host ""
if ($GpuArch -ne "unknown") {
    Write-Host "Setting up kernels for $GpuArch..."
    $kernelSrc = "$SrcDir\kernels\compiled\$GpuArch"
    $kernelDst = "$BinDir\kernels\compiled\$GpuArch"
    if (Test-Path $kernelSrc) {
        New-Item -ItemType Directory -Force -Path $kernelDst | Out-Null
        Copy-Item "$kernelSrc\*.hsaco" $kernelDst -Force
        $count = (Get-ChildItem "$kernelDst\*.hsaco").Count
        Write-Host "  Copied $count kernels ✓" -ForegroundColor Green
    } else {
        Write-Host "  No pre-compiled kernels for $GpuArch." -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping kernel setup (GPU arch unknown)."
}

# ─── Config ──────────────────────────────────────────────
$configPath = "$HipfireDir\config.json"
if (-not (Test-Path $configPath)) {
    @"
{
  "temperature": 0.3,
  "top_p": 0.8,
  "max_tokens": 512,
  "gpu_arch": "$GpuArch"
}
"@ | Set-Content $configPath
    Write-Host ""
    Write-Host "Config: $configPath"
}

# ─── PATH ────────────────────────────────────────────────
Write-Host ""
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$BinDir*") {
    $reply = Read-Host "Add hipfire to PATH? [Y/n]"
    if ($reply -ne "n") {
        [Environment]::SetEnvironmentVariable("PATH", "$BinDir;$userPath", "User")
        $env:PATH = "$BinDir;$env:PATH"
        Write-Host "  Added to user PATH ✓" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== hipfire installed ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick start:"
Write-Host "  hipfire list                        # see local models"
Write-Host "  hipfire run <model.hfq> `"Hello`"   # generate text"
Write-Host "  hipfire serve                       # start API server"
Write-Host ""
```

- [ ] **Step 2: Verify PowerShell syntax**

Run: `pwsh -NoProfile -Command "& { Get-Content scripts/install.ps1 | Out-Null; Write-Host 'Syntax OK' }"` (if pwsh available) or just review the file.

- [ ] **Step 3: Commit**

```bash
git add scripts/install.ps1
git commit -m "install: add PowerShell installer for Windows"
```

---

### Task 4: Windows CLI wrapper

**Files:**
- Create: `scripts/hipfire.cmd`

- [ ] **Step 1: Create the batch wrapper**

```batch
@echo off
bun run "%USERPROFILE%\.hipfire\cli\index.ts" %*
```

- [ ] **Step 2: Commit**

```bash
git add scripts/hipfire.cmd
git commit -m "cli: add hipfire.cmd Windows wrapper"
```

---

### Task 5: Update README with Windows instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Windows install block to Quickstart**

After the existing Linux install line, add:

```markdown
# Windows (PowerShell, requires AMD GPU)
irm https://raw.githubusercontent.com/Kaden-Schutt/hipfire/alpha-builds/scripts/install.ps1 | iex
```

- [ ] **Step 2: Update install.sh Windows detection message**

In `scripts/install.sh`, update the MINGW/MSYS/Cygwin case to reference the PowerShell installer instead of just suggesting WSL2.

- [ ] **Step 3: Commit**

```bash
git add README.md scripts/install.sh
git commit -m "docs: add Windows install instructions"
```

---

### Task 6: Test cross-compile and push

- [ ] **Step 1: Install cross target and build**

```bash
rustup target add x86_64-pc-windows-gnu
sudo apt install -y gcc-mingw-w64-x86-64
cargo build --release --target x86_64-pc-windows-gnu --features deltanet \
  --example daemon --example infer -p engine 2>&1 | tail -5
```

- [ ] **Step 2: Verify the .exe**

```bash
file target/x86_64-pc-windows-gnu/release/examples/daemon.exe
ls -lh target/x86_64-pc-windows-gnu/release/examples/daemon.exe
```

Expected: `PE32+ executable` and reasonable size (~1-2MB).

- [ ] **Step 3: Final commit and push**

```bash
git add -A
git commit -m "windows: cross-compile target, installer, CLI wrapper"
git push origin alpha-builds
```
