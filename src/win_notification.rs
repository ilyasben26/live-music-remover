use windows::{
    core::HSTRING,
    Data::Xml::Dom::XmlDocument,
    Win32::{Foundation::*, System::Registry::*},
    UI::Notifications::{ToastNotification, ToastNotificationManager},
};

const APP_ID: &str = "com.live-music-remover.portable";
const APP_NAME: &str = "Live Music Remover";
const ICON_BYTES: &[u8] = include_bytes!("../assets/logo.png");

fn extract_icon() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("{}.icon.png", APP_ID));
    if !path.exists() {
        std::fs::write(&path, ICON_BYTES).expect("Failed to write icon");
    }
    path
}

fn register_app_id(icon_path: &str) {
    let key_path = format!("Software\\Classes\\AppUserModelId\\{}", APP_ID);

    unsafe {
        let mut hkey = HKEY::default();

        let result = RegCreateKeyW(HKEY_CURRENT_USER, &HSTRING::from(key_path), &mut hkey);

        if result != ERROR_SUCCESS {
            return;
        }

        let write_str = |hkey: HKEY, name: &str, value: &str| {
            let bytes: Vec<u8> = value
                .encode_utf16()
                .chain(std::iter::once(0))
                .flat_map(|c| c.to_le_bytes())
                .collect();
            RegSetValueExW(hkey, &HSTRING::from(name), 0, REG_SZ, Some(&bytes));
        };

        write_str(hkey, "DisplayName", APP_NAME);
        write_str(hkey, "IconUri", icon_path);

        RegCloseKey(hkey);
    }
}

fn notify(title: &str, body: &str, launch_url: Option<&str>) -> windows::core::Result<()> {
    let launch_attr = match launch_url {
        Some(url) => format!(r#" launch="{}" activationType="protocol""#, url),
        None => String::new(),
    };

    let xml = format!(
        r#"<toast{}>
            <visual>
                <binding template="ToastGeneric">
                    <text>{}</text>
                    <text>{}</text>
                </binding>
            </visual>
        </toast>"#,
        launch_attr, title, body
    );

    let doc = XmlDocument::new()?;
    doc.LoadXml(&HSTRING::from(xml))?;

    let toast = ToastNotification::CreateToastNotification(&doc)?;
    let notifier = ToastNotificationManager::CreateToastNotifierWithId(&HSTRING::from(APP_ID))?;
    notifier.Show(&toast)?;

    Ok(())
}

pub fn show_update_notification(latest_version: &str, release_url: &str) {
    let icon_path = extract_icon();
    register_app_id(&icon_path.to_string_lossy());

    let body = format!(
        "Version {} is available. Click to open the download page.",
        latest_version
    );

    if let Err(e) = notify("Live Music Remover: Update Available", &body, Some(release_url)) {
        log::warn!("Failed to show update notification: {}", e);
    }
}
