# VIctory on a 10 GB NVIDIA card: a working fork and a pile of notes

This is a fork of [VIctory](https://github.com/Marvisak/VIctory), the compatibility layer that runs an Xbox Series title on Windows from files you supply yourself. I built it on the usual chain of forks (see credits) and spent most of my time measuring things on one very ordinary machine:

- Ryzen 7 5700X, 64 GB RAM
- RTX 3080 10 GB, Resizable BAR on
- Windows 11, native 1440p, no upscaler, no frame generation

I'm putting it up because of the notes, not because of the frame rate. If you want the fastest build, this isn't it.

## Why this is taking a while

The console has one pool of memory (unified memory) that the CPU and GPU share. The game writes something and the graphics chip just sees it. A PC with a separate graphics card has two pools, system memory and VRAM, so every frame the layer has to work out what the game changed, copy it to the card, and copy results back. The game never says what it changed, because on a console it never needed to. That bookkeeping is most of the frame time, not the graphics work. On my 3080 the GPU sits nearly idle.

That is also most of why numbers differ so much between people. A faster CPU helps a lot. Some AMD setups can let the GPU read the game's memory directly, which is much slower on NVIDIA. And many of the higher numbers come from rendering at a lower resolution and upscaling.

Resizable BAR might offer a way around part of it on NVIDIA. I have measured that it is fast enough in principle, but nobody has it working yet, me included. It is not impossible. It just needs more development than most people expected.

## Where it actually stands

| | |
|---|---|
| Heavy scenes | about 10 to 12 fps (85 to 100 ms a frame) |
| Light interiors | 20 to 30 fps |
| Menus, loading | up to 120 (the 30 fps cap is unlocked, game speed checked) |
| Opening cutscene | 2.5 to 6 fps in the worst shots, textures arrive late |
| Posterised/banded picture | fixed |
| Ray tracing | off. A hardware path is half built, see below |
| Mouse | works, through the right stick |

The GPU is nearly idle the whole time (1 to 2 ms a frame). Everything is CPU: finding out what the title wrote to memory and sending it to the card, bringing results back, and translating its command lists. That is the whole story of this project on a discrete card.

What does not look right yet: interior windows show a flat sky colour, eyes can look mirror white, some indoor paper glows blue or purple. My working theory is that these are all ray misses being shaded as sky, because every ray is answered "miss" while ray tracing is off. One fork thinks the purple paper is stale probe memory instead. I have not proven either.

## What is in here that you may not have seen

- **Copy-back that only merges when the title reads.** I trapped a sample of the pages copied back from the card each frame. The title read none of them. `VICTORY_BACK_ON_READ` still fetches, but merges a block into title memory only when the title touches it, with no locks on that path. It measured 5 to 8 % in back-to-back phases on its own branch. After merging it with everything else one whole-run number came out slower, so it is off by default until I have a clean A/B on the merged build. Treat it as unproven.
- **Texture queue fixes.** When textures wait for room, idle copies are freed straight away instead of waiting for the periodic sweep (`VICTORY_DEMAND_EVICT`). In the open world the number of textures due and waiting dropped to about a seventh.
- **A banding detector.** `tools/band-scan.py` scores a frame for the posterised look so you do not have to judge it by eye. I was wrong about the cause five times before I wrote it.
- **A profiler** with per-scope timings, stall and long-block lines, and in-run A/B (`VICTORY_AB=<seconds>,<SWITCH>`), which is the only way I got numbers I trust. Whole-run averages move by 15 % on their own.
- **Notes on what did not work**, with numbers: [docs/notes-3080.md](docs/notes-3080.md).
- **Ray tracing notes**, software and hardware: [docs/ray-tracing-3080.md](docs/ray-tracing-3080.md).

## Building and running

You need Rust and the Visual Studio Build Tools with the C++ workload.

```
cargo build --release -p victory
```

Then point it at your own dump (the folder with `game_scarlett_master.exe`):

```
set VICTORY_GAME_DIR=D:\path\to\your\dump
Play-VIctory.cmd
```

Switches are environment variables. The ones I run with are listed at the top of [docs/notes-3080.md](docs/notes-3080.md). `victory quality --list` prints the settings catalog.

There are no game files, screenshots or logs in this repository, and there will not be. You need your own copy of the title.

## Credits

Almost all of the code that runs here was written by other people. By line count about 70 % is the base I started from, and of what I added the largest part is an experimental shader stage that is switched off. I ported fixes from several forks and each one is named in the commit that brought it in.

- [Marvisak/VIctory](https://github.com/Marvisak/VIctory), the original
- [fxceless/victory-fork](https://github.com/fxceless/victory-fork)
- [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)
- codec-d's cool_fox
- [trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA)
- readmet3xt-lab/foxhound (the static sampler finding behind the banding fix)
- TaynanGT/Vice-Tay, Peppacodes/playablegames-peppa, ItsMK1/VIctory-MnK-patches, maren-halv/VIctory-DXR, aleksesipenko/VIctory-RTX, Belobug, DexNom and the others listed in [docs/README-base-fork.md](docs/README-base-fork.md), which is the README of the fork this started from, kept as it was.

XCL (GPL-2.0-only) was read for ideas only. None of its code is here.

## License

GPL-3.0, same as upstream. See [LICENSE](LICENSE).
