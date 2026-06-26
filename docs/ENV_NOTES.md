# Environment notes & open blockers (2026-06-25)

Two environment problems were found while resuming the Sionna passive-radar
work. Neither blocks computation (GPU ray tracing works), but both need an
**admin / root** action to put files where PROJECT_CONTEXT sec.11 wants them.

## 0. OptiX — RESOLVED (was the "미해결 이슈" in PROJECT_CONTEXT sec.10)
`rt.load_scene(...)` + `PathSolver` now run on GPU. The OptiX driver libs are
present and loadable:
- `/lib/x86_64-linux-gnu/libnvoptix.so.1` (+ `libnvidia-rtcore.so`) are mounted
  and registered in `ldconfig`; `ctypes.CDLL("libnvoptix.so.1")` succeeds.
- Mitsuba `cuda_ad_mono_polarized` renders; Sionna `PathSolver` traces paths.

Root cause of the *original* failure (for the record): `libnvoptix.so.1` /
`libnvidia-rtcore.so` belong to the NVIDIA container's **graphics** capability
group, not the default `compute,utility`. The container was previously started
without them; it has since been (re)started with the graphics/all caps, so they
are now bind-mounted. **No `DRJIT_LIBOPTIX_PATH` override is needed.**
If it ever regresses: ask the admin to start the container with
`NVIDIA_DRIVER_CAPABILITIES=all` (or `compute,utility,graphics`).

## 1. `/workspace` symlink is missing -> `conda activate` is broken
The conda env is installed at `/home/yunjung/workspace/jeong/miniforge3`, but it
is configured for the path `/workspace/jeong/miniforge3` (hardcoded in
`etc/profile.d/conda.sh`). `/workspace` does not exist in this container, so
`conda activate sionna` fails with "No such file or directory".

- **Workaround (no admin needed):** call the env python by absolute path —
  `/home/yunjung/workspace/jeong/miniforge3/envs/sionna/bin/python`
  (this is what `run.sh` does). All packages import fine this way.
- **Proper fix (admin/root):** recreate the symlink so every hardcoded path and
  muscle-memory command works again:
  `sudo ln -s /home/yunjung/workspace /workspace`

## 2. Permissions — this session runs as `yunjung`, not root
PROJECT_CONTEXT assumes root. This session is uid 1015 `yunjung`
(groups: yunjung, irs_open1) with **no passwordless sudo**, and cannot write to
any of the designated project dirs:

| Path                         | Owner:Group   | yunjung access |
|------------------------------|---------------|----------------|
| `/workspace/jeong` (code)    | root:root 755 | read-only      |
| `/data/public/jeong` (data)  | root:member   | read-only (not in `member`) |
| `/data/ckpoint/jeong` (ckpt) | root:root 755 | read-only      |
| `/home/yunjung/workspace`    | yunjung       | **writable**   |

Because of this, code + the first RD map were staged under
`/home/yunjung/workspace/jeong_sionna/` (writable). **This is temporary.**

### Admin fix (pick one)
- add `yunjung` to the `member` group and `chmod g+w` the dirs, **or**
- `chown -R yunjung:yunjung /home/yunjung/workspace/jeong` and give `yunjung`
  write on `/data/public/jeong` + `/data/ckpoint/jeong`, **or**
- run the Claude Code session as root.

### Relocation once writable (run as a user who can write the targets)
```bash
# code -> /workspace/jeong  (SSD)
cp -r /home/yunjung/workspace/jeong_sionna /workspace/jeong/sionna_radar
# results -> /data/public/jeong  (HDD)
mkdir -p /data/public/jeong/sionna/stage1
cp /home/yunjung/workspace/jeong_sionna/outputs/* /data/public/jeong/sionna/stage1/
```
Then run with the proper output dir directly:
```bash
./run.sh --outdir /data/public/jeong/sionna/stage1
```
(`passive_radar_stage1.py` already auto-targets `/data/public/jeong/sionna/stage1`
when it is writable; otherwise it falls back to `./outputs`.)
