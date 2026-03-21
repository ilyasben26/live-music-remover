

```powershell
$Env:RUST_LOG="live-_music_remover=INFO,live_music_remover=DEBUG,df=DEBUG"
$Env:DF_MODEL="c:\Users\ilyas\Repos\DeepFilterNet\models\DeepFilterNet3_ll_onnx.tar.gz"
cargo +nightly run -p live-music-remover --features ui --bin live-music-remover --release
cargo +nightly build -p live-music-remover --features ui --bin live-music-remover --release
cargo +nightly build -p live-music-remover --features "ui,dev-console" --bin live-music-remover --release
```