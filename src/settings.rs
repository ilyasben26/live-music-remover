//! User settings persisted as JSON in the local app data folder, next to the
//! update checker's files.

use std::fs;
use std::path::PathBuf;

/// Bump when the EULA text changes so users are asked to accept it again.
pub const EULA_VERSION: u64 = 1;

pub const EULA_TEXT: &str = "\
Please read this before using Live Music Remover (\"the Software\"). By clicking \
\"I accept\", you agree to these terms. If you don't agree, click \"Exit\" and the \
Software will close.

1. License
The Software is free and open source under the MIT License, © 2026 Ilyas Benyamna. \
You may use, copy, modify and distribute it under that license. The full text is in \
the LICENSE file and at https://github.com/ilyasben26/live-music-remover.

2. No warranty
The Software is provided \"as is\", without any warranty, express or implied. Music \
and noise removal is not perfect: short bursts of music, artifacts or distortion may \
get through, and speech may sometimes be degraded. The Software is a work in progress \
and may contain bugs.

3. Limitation of liability
The author is not liable for any damages arising from use of the Software. This \
includes, but is not limited to, data loss, system instability, hearing discomfort \
from unexpected audio levels, and any consequences of relying on the Software's output.

4. Your responsibility
You are solely responsible for how you use the Software and for any audio you process \
with it. Keep your volume at a safe level.

5. Privacy
All audio processing happens locally on your device. No audio is recorded, stored or \
sent anywhere. If update checking is turned on, the Software contacts the GitHub API \
(api.github.com) about once a day to see if a new version is available, which lets \
GitHub see your IP address as part of a normal web request. You can turn update \
checking off below or at any time on the Settings page. No other data is collected.

6. Third-party components
The Software includes third-party models and libraries under their own licenses: \
DeepFilterNet (MIT/Apache-2.0), DPDFNet (Apache-2.0), Resemble Enhance (MIT, © 2023 \
Resemble AI) and ONNX Runtime (MIT). VB-CABLE, if you install it, is separate software \
from VB-Audio with its own terms.

7. Changes
These terms may be updated in future versions. If they change, you'll be asked to \
accept them again.";

pub struct Settings {
    /// Version of the EULA the user accepted, 0 if never.
    pub eula_accepted_version: u64,
    pub check_for_updates: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            eula_accepted_version: 0,
            check_for_updates: true,
        }
    }
}

fn settings_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("live-music-remover").join("settings.json"))
}

impl Settings {
    pub fn eula_accepted(&self) -> bool {
        self.eula_accepted_version >= EULA_VERSION
    }

    pub fn load() -> Self {
        let mut settings = Self::default();
        let Some(json) = settings_path()
            .and_then(|p| fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        else {
            return settings;
        };
        if let Some(v) = json["eula_accepted_version"].as_u64() {
            settings.eula_accepted_version = v;
        }
        if let Some(v) = json["check_for_updates"].as_bool() {
            settings.check_for_updates = v;
        }
        settings
    }

    pub fn save(&self) {
        let Some(path) = settings_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let json = serde_json::json!({
            "eula_accepted_version": self.eula_accepted_version,
            "check_for_updates": self.check_for_updates,
        });
        if let Err(e) = fs::write(&path, json.to_string()) {
            log::warn!("Failed to save settings: {e}");
        }
    }
}
