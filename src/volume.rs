use windows::Win32::Media::Audio::Endpoints::IAudioEndpointVolume;
use windows::Win32::Media::Audio::{eConsole, eRender, IMMDeviceEnumerator, MMDeviceEnumerator};
use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED};

/// Wraps the Windows default render endpoint volume so the worker thread can
/// read and apply the system master volume (including mute) to every frame.
pub struct SystemVolume {
    endpoint_volume: IAudioEndpointVolume,
}

impl SystemVolume {
    /// Initialises COM for the calling thread and opens the default render
    /// endpoint.  Should be called from the thread that will use it.
    pub fn new() -> anyhow::Result<Self> {
        unsafe {
            // CoInitializeEx is idempotent; RPC_E_CHANGED_MODE just means
            // another apartment type is already active on this thread, which
            // is fine — we can still use COM objects.
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
            let endpoint_volume: IAudioEndpointVolume =
                device.Activate(CLSCTX_ALL, None)?;

            Ok(Self { endpoint_volume })
        }
    }

    /// Returns the effective gain scalar to apply to each sample.
    /// Returns 0.0 when the endpoint is muted, otherwise the master volume
    /// level in [0.0, 1.0].
    pub fn get_scalar(&self) -> f32 {
        unsafe {
            let muted = self.endpoint_volume.GetMute().unwrap_or_default();
            if muted.as_bool() {
                return 0.0;
            }
            self.endpoint_volume
                .GetMasterVolumeLevelScalar()
                .unwrap_or(1.0)
        }
    }
}
