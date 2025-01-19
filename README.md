# openseeface-rs

`openseeface-rs` is a port of the popular [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace)
tracking library and daemon to Rust. 🦀

This library works very similarly to the original OpenSeeFace, which has
excellent documentation.

### ⚠️ This is not a production-ready library ⚠️

This library is just a hobby project. Do not use it in critical situations.
I can heartily recommend the original [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace).

#### Known Issues / Missing Features
- RetinaFace tracking support

## Usage

To use `openseeface-rs` as an OpenSeeFace alternative, run the tracker binary.
It supports a few of the options from regular OpenSeeFace at the moment:

    cargo run --release --package=openseeface-tracker -- --model=3 --address=127.0.0.1:11573

You can list available video devices with `--list-cameras`.

## Demo

The demo package includes a frontend built in Bevy to debug the tracker. The
demo takes similar command line arguments to the tracker.

You can toggle various overlays in the demo:
  - F2 - Toggle image-space landmarks overlay
  - F3 - Toggle reference face mesh overlay with best fit transform
  - F4 - Toggle output 3D face mesh overlay
  - F6 - Toggle capturing new frames
  - F7 - Toggle Bevy entity inspector
  - F8 - Toggle Bevy resource
  - F10 - Dump meshes to an .obj file
  - F11 - Dump intermediate textures to EXR files

## As a Library

`openseeface-rs` is designed to be embedded in other applications too. All
required models are embedded at build time.

Usage is straight-forward, just provide the tracker with each video frame as
it is available.

```rust
let mut tracker = Tracker::new(TrackerConfig::default())?;
tracker.detect(&image, now)?;

for face in tracker.faces() {
    // Use visible faces...
}
```

## Thanks

The models and many of the algorithms used in this library were conceived by
those who contributed to [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace/tree/master/models),
particularly [@emilianavt](https://github.com/emilianavt). I'm very grateful
for all the hard work that has gone into OpenSeeFace!

## License

The code portion of this library is permissively dual licensed under the
[MIT license](LICENSE-MIT) and [Apache 2.0 license](LICENSE-APACHE) licenses.

The models used for this library are the same ones from the original
[OpenSeeFace](https://github.com/emilianavt/OpenSeeFace/tree/master/models) and
fall under their original [BSD-2 terms](models/LICENSE).
