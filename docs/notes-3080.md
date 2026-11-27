# Notes from a 3080 10 GB

Everything here was measured on a Ryzen 7 5700X with an RTX 3080 10 GB at native 1440p. Numbers are per frame in a heavy scene unless I say otherwise. Where I only have one run I say so.

A word on method, because it cost me days: whole-run averages are useless for anything under 15 %. Other work on the machine moves them more than most changes do. Everything I trust came from flipping one switch on and off in phases inside a single run (`VICTORY_AB=20,SWITCH_NAME`) and comparing neighbouring phases.

## The settings I run with

```
VICTORY_RELAY=2 VICTORY_CARD_STEPS=2 VICTORY_WIDE_BACK=1 VICTORY_CARD_REHASH_RUNS=1
VICTORY_JOIN_COPIES=1 VICTORY_INDIRECT_WINDOW=1 VICTORY_RELAY_COPY=1 VICTORY_INDIRECT_BATCH=2
VICTORY_MAP_THREADS=1 VICTORY_TITLE_PRIORITIES=0 VICTORY_CARD_AWAKE=1 VICTORY_SHOW_AFTER=1
VICTORY_THIN=1 VICTORY_JOIN_PARTS=8 VICTORY_UPLOAD_AHEAD=1 VICTORY_NAN_SCRUB=1
VICTORY_AMBIENT=0.8 VICTORY_AMBIENT_AIR=1.5 VICTORY_AMBIENT_MOST=2 VICTORY_AMBIENT_MAGENTA=0
VICTORY_DYNAMIC_SLOTS=524288 VICTORY_CARD_BUFFERS_MIN_KB=64 VICTORY_CARD_BUFFERS_MB=2048
VICTORY_BVH_MISS=1 VICTORY_RESIDENCY=legacy VICTORY_EXACT64=0 VICTORY_OPEN_OUTSIDE=1
VICTORY_DISPLAY_MODE=2560x1440 VICTORY_TEXTURE_FREE_UNMAPPED=1 VICTORY_MERGE_SUBMISSIONS=1
VICTORY_FPS=120
```

## Where a frame goes

About 85 to 100 ms:

| | ms |
|---|---|
| Finding what the title wrote and uploading it | about 45 |
| Translating the title's command lists | about 20 |
| Copying GPU results back | 20 to 30 |
| The GPU itself | 1 to 2 |

Per frame the runtime asks "did this copy change?" about 5,900 times over 5.2 GB. About 5,000 answers are "no" and cost around 8 ms. About 650 really changed and 76 to 98 MB gets copied. Windows is asked about 1.4 GB of pages and reports 28 MB written.

## Things that worked

**Posterised picture.** foxhound found the cause; I only confirmed it. A pipeline with static samplers that is loaded from the D3D12 pipeline library can sample through the wrong sampler on NVIDIA. The composite pass then reads its 33x33x33 grading tables unfiltered. The fix is to never load or store such pipelines. The state lives in the library file, so it is sticky per entry: a build can be banded in every run or in none, and a "fresh cache" run still bands on the second launch. Controls only mean something on a run that loads what an earlier run stored. Cost: roughly half of about 2,350 pipelines are compiled again each launch, which I could not see in the frame rate. About 23 distinct static samplers are in use, so creating them in a fixed order at start-up should let the library keep everything. I wrote that and never ran it, so it is not in here.

**Merging submissions** (from playablegames-peppa): 5 to 10 ms.

**Small buffers on the card** (`VICTORY_CARD_BUFFERS_MIN_KB=64`): 135 to 110 ms at the time. Anything the GPU touches in system memory costs a bus round trip per access.

**Freeing textures the title has unmapped**, and demand-driven eviction when textures wait for room. In the open world the queue of due textures went from about 1,400 to about 200 and the oldest waiting texture from 23 frames to 2. Single runs.

**The title's own watchdog.** It kills the process when its main thread has not advanced for about 30 seconds, which a slow load triggers with nothing actually wrong. It looks like a null write inside the game image. Patched out, byte-checked, with an off switch.

**Frame rate unlock.** Three patch sites from the 7900XTX fork. The title's own clock stays at wall speed at 60, 120 and 294 fps. Its menu logic still ticks 30 times a second by design.

**Mouse.** The title never asks GameInput for a mouse: zero registrations and zero reads with a device present and moving. It reads keyboard and gamepad only. Mouse has to drive the right stick.

## Things that did not

| Idea | Result |
|---|---|
| GPU reads title memory in place (zero copy) | about 100 ms slower. Random GPU reads of system memory measured 26x slower than card memory, 14 ns per access |
| The lineage's in-place renderer ports | slow, or a crash at world start |
| The lineage's newer base as a whole | 3x slower here. One commit gives every misaligned structured view its own copy buffer on hosts without raw twins. With that off it matches |
| Delta upload for buffers shaders never write | only 1.7 MB a frame qualifies |
| Delta upload per shader-written range | only 8 to 10 MB qualifies |
| Parallel "nothing changed" answers | about 1 ms |
| Spinning on the guards lock | no change. The waits are real holds across VirtualProtect, which Windows serialises per process |
| Fused list planning, skip covered refreshes | no change |
| Helper threads at the submitting thread's priority | no change on 8 cores (it helped its author on 6) |
| Read-backs on their own thread | at most 1 ms here |
| A wider landing pool | no change |
| Software ray tracing | see the other document |

The one about uploads deserves a sentence. Only about 24 % of the bytes copied up each frame differ. You cannot send just the difference, because the big buffers are written by both the title and by shaders on the card, and the CPU snapshot does not have the shader-written part. When that was tried without the restriction a room lost its indirect light and geometry exploded.

## Copy-back

A probe made one in sixteen landed 64 KiB pages inaccessible and counted what the title did. Over a run it read none and wrote fourteen, all in one buffer. 53 of about 88 MB a frame sits in copies it never touched.

Two mechanisms follow from that:

1. **Deferred merge** (in this tree, `VICTORY_BACK_ON_READ` with `_LOOSE` and `_PRIVATE`, off by default). Fetch as before into a stash, merge a block only when the title reads or writes it. The fault path does a compare-and-swap on a per-block state and copies from the stash: no graphics calls, no allocation, no locks. An earlier version resolved through the normal landing code and deadlocked on its own locks. Merge time went from about 20 to about 10 ms and the frame 5 to 8 % in phase A/B on its branch. The first whole run after merging into this tree was slower, so I do not yet claim it.
2. **Skip the fetch entirely** for big buffers nobody reads (from maren-halv's fork). 8.6 ms in six of six phase pairs. Not in this tree: as written, a direct read by the title is invisible and would get stale bytes. It needs the same read trap as above before it is safe.

## Why cutscenes load worse than gameplay

Not pipelines. A pool of ten workers finishes all of them before the first world frame. The log's "being made" count misled me because it includes about a thousand ray tracing pipelines that are left out on purpose.

Each camera cut remaps and rebinds a whole scene while the card is still full from the previous shot:

- Textures queue at the admission line for about a minute with 100 to 400 MB missing while 700 to 900 MB of the card is free. Raising the line briefly when textures are stuck cut missing-texture time by 88 % in one pair of runs. That change is not merged yet.
- The title's threads spend 15 to 70 times longer in the memory mapping calls than in the world: over 4,000 per-page map calls a second, 160 to 300 microseconds each, about a millisecond when the address space lock is contended.

## The idea I have not finished

With Resizable BAR on, D3D12 has a heap that is CPU-writable and lives in card memory. Measured here:

| | |
|---|---|
| GPU reads of it against normal card memory | 0.54 against 0.52 ns per access |
| CPU writes into it | about 20 GB/s |
| CPU reads from it | 12 to 24 MB/s |
| Bulk copy system memory to card | 30 MB in 1.2 ms, 200 MB in 8.3 ms |

The title marks its big per-frame allocations write-combined itself, which is the contract that memory needs. If the hot part of the title's per-frame memory were backed by it, most of the tracking and uploading would have nothing to do. The open problem is placing that mapping at the addresses the title uses, or handing it out when the title allocates. My first test of forcing the address was wrong (I constrained the address space after the driver had already chosen), so this is undecided, not disproven. Only the hot subset could ever fit: those allocations total about 8 GB.

If you know this corner of D3D12 and Windows memory mapping, this is the part I would most like help with.
