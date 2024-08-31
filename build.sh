#!/usr/bin/env bash

# see: https://github.com/rustwasm/wasm-pack/issues/811#issuecomment-950013885

cargo build --lib --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir static target/wasm32-unknown-unknown/release/wasm_lbm.wasm
wasm-opt static/wasm_lbm_bg.wasm -o static/wasm_lbm_bg.wasm -O4