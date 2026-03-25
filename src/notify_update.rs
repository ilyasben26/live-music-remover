use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const GITHUB_API_URL: &str =
    "https://api.github.com/repos/ilyasben26/live-music-remover/releases/latest";
const CHECK_INTERVAL_SECS: u64 = 86400; // 24 hours

fn data_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("live-music-remover"))
}

fn last_check_path() -> Option<PathBuf> {
    data_dir().map(|d| d.join("last_update_check"))
}

fn last_known_release_path() -> Option<PathBuf> {
    data_dir().map(|d| d.join("last_known_release"))
}

/// Persist the latest known release (tag\nurl) to disk.
fn save_known_release(tag: &str, url: &str) {
    let path = match last_known_release_path() {
        Some(p) => p,
        None => return,
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&path, format!("{}\n{}", tag, url));
}

/// Remove the persisted release file (user is already on latest).
fn clear_known_release() {
    if let Some(path) = last_known_release_path() {
        let _ = fs::remove_file(path);
    }
}

/// Load a previously saved (tag, url) pair from disk.
fn load_known_release() -> Option<(String, String)> {
    let path = last_known_release_path()?;
    let content = fs::read_to_string(path).ok()?;
    let mut lines = content.splitn(2, '\n');
    let tag = lines.next()?.trim().to_string();
    let url = lines.next()?.trim().to_string();
    if tag.is_empty() || url.is_empty() {
        return None;
    }
    Some((tag, url))
}

fn should_check() -> bool {
    let path = match last_check_path() {
        Some(p) => p,
        None => return true,
    };

    let Ok(content) = fs::read_to_string(&path) else {
        return true;
    };

    let Ok(last_check): Result<u64, _> = content.trim().parse() else {
        return true;
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    now.saturating_sub(last_check) >= CHECK_INTERVAL_SECS
}

fn record_check() {
    let path = match last_check_path() {
        Some(p) => p,
        None => return,
    };

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let _ = fs::write(&path, now.to_string());
}

fn parse_version(v: &str) -> Option<(u32, u32, u32)> {
    let v = v.trim_start_matches('v');
    let mut parts = v.splitn(3, '.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    Some((major, minor, patch))
}

fn is_newer(latest: &str, current: &str) -> bool {
    match (parse_version(latest), parse_version(current)) {
        (Some(l), Some(c)) => l > c,
        _ => false,
    }
}

pub fn check_for_update(update_info: Arc<Mutex<Option<(String, String)>>>) {
    let current = env!("CARGO_PKG_VERSION");

    // On every startup, restore persisted release info — but only if it's still
    // newer than the running binary (handles the case where the user downloaded
    // the update and is now running the new version).
    if let Some((saved_tag, saved_url)) = load_known_release() {
        if is_newer(&saved_tag, current) {
            if let Ok(mut info) = update_info.lock() {
                *info = Some((saved_tag, saved_url));
            }
        } else {
            // Current version >= saved latest: user has updated, clear the file.
            log::info!("Running version {} >= saved latest, clearing update cache", current);
            clear_known_release();
        }
    }

    if !should_check() {
        return;
    }

    record_check();

    let response = match ureq::get(GITHUB_API_URL)
        .set("User-Agent", "live-music-remover")
        .call()
    {
        Ok(r) => r,
        Err(e) => {
            log::warn!("Update check failed: {}", e);
            return;
        }
    };

    let body = match response.into_string() {
        Ok(b) => b,
        Err(e) => {
            log::warn!("Failed to read update response: {}", e);
            return;
        }
    };

    let json: serde_json::Value = match serde_json::from_str(&body) {
        Ok(j) => j,
        Err(e) => {
            log::warn!("Failed to parse update response: {}", e);
            return;
        }
    };

    let latest_tag = match json["tag_name"].as_str() {
        Some(t) => t,
        None => return,
    };

    let html_url = json["html_url"].as_str().unwrap_or(
        "https://github.com/ilyasben26/live-music-remover/releases/latest",
    );

    log::info!("Update check: current={}, latest={}", current, latest_tag);

    if is_newer(latest_tag, current) {
        save_known_release(latest_tag, html_url);
        if let Ok(mut info) = update_info.lock() {
            *info = Some((latest_tag.to_string(), html_url.to_string()));
        }
        crate::win_notification::show_update_notification(latest_tag, html_url);
    } else {
        // No newer release — ensure any stale cache is cleared.
        clear_known_release();
    }
}
