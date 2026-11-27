# Ray tracing on an NVIDIA card: what I tried and where it is

Short version: ray tracing is off in this tree. The software route works and is far too slow. The hardware route has one shader running on the card and its hits are not right yet. Neither gives a picture I would show anyone.

## Why it matters at all

The title traces rays for indoor light, reflections, glass and some shadows. With ray tracing off, every ray is answered "miss", and the title shades a miss with the sky. I think that is behind several things that look wrong:

- interior windows showing a flat sunset colour instead of the room behind (windows facing outdoors look right, because the sky really is there)
- eyes that look like chrome
- glossy surfaces glowing
- paper indoors lit blue or purple

That is a theory. The purple paper has a competing explanation from another fork (stale probe memory the title never rewrote). I do not have a rays-on picture good enough to settle it.

The tree carries stand-ins that soften this while rays are off (`VICTORY_AMBIENT`, `VICTORY_AMBIENT_AIR`, `VICTORY_AMBIENT_MOST`, and a magenta guard that is off by default). They are estimates, not the title's lighting. They should all go off the day real rays work.

## How the title does it

The console has no DXR. The title's shaders walk a bounding volume hierarchy in the console's own format (box nodes with 4 children in 32 or 16 bit precision, triangle nodes, instance nodes) with one GPU instruction per step. The structures are built by a console library the title ships with. 64 of its shader programs contain such a walk.

## Route 1: software (in this tree, off)

About 3,100 lines: `crates/victory-shader/src/bvh_build.rs`, `crates/victory-hle/src/modules/bvhbuilder.rs`, and the lowering of the intersect instruction in `crates/victory-shader/src/emit/intersect.rs` (that last one is mostly inherited).

What I did:

- The console's builder library hung or faulted on this machine for some inputs, so I wrote a builder that produces the same node format on the CPU, with a validator.
- A budget: each walking program gets a number of node tests per frame, adapted to how long the card was waited for, with a target of 80 ms. Rays over budget are answered as misses. A zero budget costs nothing.
- An earlier hang turned out to be one class of walks that had no budget at all, over a 182 MB structure buffer.

Why it is a dead end on this hardware:

- **Building.** The title rebuilds animated geometry every frame into fresh buffers, about 115 full builds a second of about 3,600 triangles each. The tree is 91 % of the cost at about 1.2 microseconds per triangle. That is around 100 ms of CPU a frame. A cache keyed on input and content hit 17 %. Even refitting everything would be 30 to 50 ms.
- **Walking.** 20 to 165 ns per node test, where the card should manage well under one. Three programs take nearly all of it: the screen probe trace and two that trace the visibility buffer. Structures were already in card memory, and the smaller node format made no difference.
- **Result.** Rays on costs 65 to 130 ms a frame on top of a 100 ms frame, and inside that budget most rays still miss. The room got darker and noisier, not better.

One mistake worth recording: I thought rebuilding the title's shipped static structures with my builder cut GPU time by 3.3x. It did not. Those meshes store positions in a packed two-component format that only the console library decodes. I had read them as flat points, so the "faster" trees were tracing garbage.

## Route 2: hardware ray queries (not in the main tree)

The work is in [`wip/hardware-ray-tracing.diff`](../wip/hardware-ray-tracing.diff), about 13,000 added lines against an earlier commit of this tree. Roughly 1,200 of those are mine (`dxr.rs`, the sidecar that builds real acceleration structures). The shader side, which rewrites the title's walk loops into SM 6.6 RayQuery, is adapted from aleksesipenko/VIctory-RTX under GPL-3 with attribution in the files. It is behind `VICTORY_RT=dxr` and `VICTORY_RT_LIFT`.

How it works: the console library still runs. Its calls are captured and the same geometry is built into D3D12 bottom and top level structures on the card, batched into one command list per frame on a worker thread.

What works:

- Structures for an indoor scene take about 0.3 GB.
- Building costs about 6 ms of card time a frame and no visible frame time. (The first version submitted every build separately on the title's threads and cost 90 ms. Pooling the result buffers took the card from 11.8 GB back to 9.7.)
- 46 of the 64 programs compile to RayQuery and pass DXIL validation. That matches what VIctory-RTX reports.
- One program, the indoor screen probe trace, runs live on the card without hanging.

What does not:

- **Hits do not agree with the software walker.** A check traces about a thousand rays both ways. In the last run 2,585 of 4,320 rays that hit on both sides disagreed.
- **Packed meshes.** Dynamic geometry declared as two 32 bit floats per vertex is packed data. Read as floats it gives triangles with bounds around 1e37, which every ray hits first. Those meshes are now left out, which means characters are invisible to rays. The decode is not done. There are 40 raw samples to fit it against the console library's output.
- **Instance masks.** The software walker used for checking ignores them and often reports a tiny proxy box or quad as its closest hit, while the card hits real geometry. So the reference may be the wrong side here. Not confirmed by brute force yet.
- **My own bugs along the way:** a cleanup that threw away structures the title builds once and uses forever, and triangles with invalid corners being handed to the card and registering as hits at the ray origin. Both fixed.
- **Card memory.** Turning it on costs about 550 MB before any structure exists, and holding 288 MB of structures raised usage by over a gigabyte. On a 10 GB card that comes out of textures.
- **One expensive program is not covered.** The primary visibility buffer gather has a pattern before each trace that the rewrite rejects.
- **18 programs are not covered at all**, for constructs the translator does not support.
- No timing of the rewritten shader on its own, and no trustworthy picture.

## How close is it to the real thing

Not close. With rays off the lighting indoors is an estimate and reflections fall back to sky. With software rays it is slower and looks worse. With hardware rays one of 64 programs runs and it is not yet tracing the right scene.

What I would do next, in order:

1. Make the check trace both sides with the mask the rewritten program itself uses, then brute force a few mismatching rays to learn which side is right.
2. Decode the packed vertex format, so characters exist for rays again.
3. Capture the same frames with rays off and on, with the stand-ins off, starting with the indoor probe light.
4. Turn on the reflection and transparency programs, which are the ones that should fix glass and eyes.
5. Find where the extra gigabyte of card memory goes.

If you are working on the same thing: check the vertex format before you trust a build, and check instance masks before you trust a mismatch.
